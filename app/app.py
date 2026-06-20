"""
app/app.py
-----------
RevenueIQ — Sales Analytics & ML Dashboard.

Combines a traditional BI dashboard (KPIs, trends, regional/category
breakdowns) with three integrated ML capabilities:
  - Revenue forecasting (Gradient Boosting Regressor)
  - Product/region segmentation (K-Means)
  - Order-level anomaly detection (Isolation Forest)

Run locally:
    streamlit run app/app.py
"""

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))      # .../revenueiq/app
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)                  # .../revenueiq

for path in (_PROJECT_ROOT, _THIS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import streamlit as st
import pandas as pd
import plotly.express as px

from utils import (
    load_data, validate_schema, kpi_card, to_csv_bytes, CUSTOM_CSS,
    REQUIRED_COLUMNS, standardize_columns
)
from src.features.build_features import (
    build_forecasting_features, build_product_features,
    build_customer_like_features, add_order_level_anomaly_features
)
from src.models.forecast_model import SalesForecaster
from src.models.segmentation_model import SegmentationModel, run_segmentation
from src.models.anomaly_model import AnomalyDetector

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="RevenueIQ | Sales Analytics & ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown("""
<h2 style="margin-bottom:0;">📊 RevenueIQ <span class="ml-badge">ML-POWERED</span></h2>
<p style="color:gray; margin-top:0; margin-bottom:10px;">
Sales analytics dashboard with built-in forecasting, segmentation, and anomaly detection.
</p>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SIDEBAR — UPLOAD
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📁 Data Source")
    uploaded_file = st.file_uploader("Upload Sales CSV", type=["csv"])
    use_sample = st.checkbox("Use bundled sample dataset", value=False,
                              help="Loads data/raw/sales_data.csv if no file is uploaded")

    st.markdown("---")
    st.markdown("### ⚙️ Global Filters")
    st.caption("Filters here apply across every tab.")

# ----------------------------------------------------------------------------
# DATA RESOLUTION
# ----------------------------------------------------------------------------
SAMPLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "sales_data.csv")

df_raw = None
if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
elif use_sample and os.path.exists(SAMPLE_PATH):
    df_raw = load_data(SAMPLE_PATH)

if df_raw is not None:
    missing_cols = validate_schema(df_raw)
    if missing_cols:
        st.warning(
            f"Couldn't automatically match {len(missing_cols)} required column(s): "
            f"{', '.join(missing_cols)}. Map them manually below, or fix your CSV headers and re-upload."
        )

        with st.expander("🔍 Columns detected in your file", expanded=True):
            st.markdown(f"`{', '.join(str(c) for c in df_raw.columns)}`")

        st.markdown("#### 🔧 Manual Column Mapping")
        st.caption("Pick which column in your file corresponds to each required field.")

        available_cols = ["— none —"] + list(df_raw.columns)
        col_map = {}
        map_cols = st.columns(3)
        for i, req_col in enumerate(missing_cols):
            with map_cols[i % 3]:
                col_map[req_col] = st.selectbox(req_col, available_cols, key=f"map_{req_col}")

        apply_mapping = st.button("✅ Apply Mapping & Continue")

        if not apply_mapping:
            st.stop()

        rename_dict = {v: k for k, v in col_map.items() if v != "— none —"}
        df_raw = df_raw.rename(columns=rename_dict)

        still_missing = validate_schema(df_raw)
        if still_missing:
            st.error(f"Still missing after mapping: {', '.join(still_missing)}. "
                      f"Please select a source column for each required field.")
            st.stop()

        # Re-apply type coercion now that columns have their canonical names
        # (load_data() ran this before the rename, so Date/Total Price need
        # to be redone against the newly-mapped columns).
        if "Date" in df_raw.columns:
            df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")
        for numeric_col in ["Quantity", "Unit Price", "Total Price"]:
            if numeric_col in df_raw.columns:
                df_raw[numeric_col] = pd.to_numeric(df_raw[numeric_col], errors="coerce")
        if df_raw["Total Price"].isna().all() and "Quantity" in df_raw.columns and "Unit Price" in df_raw.columns:
            df_raw["Total Price"] = df_raw["Quantity"] * df_raw["Unit Price"]

        st.success("Columns mapped successfully!")

    has_date = df_raw["Date"].notna().any()

    # ---- Sidebar global filters ----
    with st.sidebar:
        categories = sorted(df_raw["Category"].dropna().unique().tolist())
        sel_categories = st.multiselect("Category", categories, default=categories)

        regions = sorted(df_raw["Region"].dropna().unique().tolist())
        sel_regions = st.multiselect("Region", regions, default=regions)

        payment_methods = sorted(df_raw["Payment Method"].dropna().unique().tolist())
        sel_payments = st.multiselect("Payment Method", payment_methods, default=payment_methods)

        if has_date:
            min_d, max_d = df_raw["Date"].min(), df_raw["Date"].max()
            date_range = st.date_input("Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        else:
            date_range = None

        st.markdown("---")
        st.caption(f"Rows loaded: **{len(df_raw):,}**")

    # ---- Apply global filters ----
    df = df_raw[
        df_raw["Category"].isin(sel_categories) &
        df_raw["Region"].isin(sel_regions) &
        df_raw["Payment Method"].isin(sel_payments)
    ].copy()

    if has_date and date_range and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df["Date"] >= start) & (df["Date"] <= end)]

    if df.empty:
        st.warning("No data matches the current filters. Adjust filters in the sidebar.")
        st.stop()

    # ---- Tabs ----
    menu = st.radio(
        "Navigation",
        ["🏠 Overview", "📈 Trends", "🌍 Region-wise", "🏷️ Category", "🏆 Top Products",
         "💳 Payment Methods", "🔮 Forecast", "🧩 Segmentation", "🚨 Anomaly Detection",
         "🔍 Deep Filter", "📋 Raw Data"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # ========================================================================
    # OVERVIEW
    # ========================================================================
    if menu == "🏠 Overview":
        total_revenue = df["Total Price"].sum()
        total_units = df["Quantity"].sum()
        avg_order_value = df["Total Price"].mean()
        num_orders = df["Order ID"].nunique()
        best_product = df.groupby("Product Name")["Quantity"].sum().idxmax()
        highest_revenue_product = df.groupby("Product Name")["Total Price"].sum().idxmax()

        k1, k2, k3, k4 = st.columns(4)
        kpi_card("Total Revenue", f"₹ {total_revenue:,.0f}", k1)
        kpi_card("Units Sold", f"{total_units:,.0f}", k2)
        kpi_card("Avg Order Value", f"₹ {avg_order_value:,.0f}", k3)
        kpi_card("Total Orders", f"{num_orders:,}", k4)

        k5, k6 = st.columns(2)
        kpi_card("Best Selling Product", best_product, k5)
        kpi_card("Highest Revenue Product", highest_revenue_product, k6)

        st.markdown('<div class="section-title">Revenue by Category</div>', unsafe_allow_html=True)
        cat_rev = df.groupby("Category")["Total Price"].sum().reset_index().sort_values("Total Price", ascending=False)
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(cat_rev, x="Category", y="Total Price", color="Category", text_auto=".2s")
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10), height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.pie(cat_rev, names="Category", values="Total Price", hole=0.55)
            fig2.update_layout(margin=dict(t=10, b=10), height=350)
            st.plotly_chart(fig2, use_container_width=True)

    # ========================================================================
    # TRENDS
    # ========================================================================
    elif menu == "📈 Trends":
        if not has_date:
            st.info("No valid 'Date' column found — trend analysis needs dated records.")
        else:
            st.markdown('<div class="section-title">Revenue Over Time</div>', unsafe_allow_html=True)
            granularity = st.select_slider("Granularity", options=["Day", "Week", "Month"], value="Day")
            freq_map = {"Day": "D", "Week": "W", "Month": "ME"}
            ts = df.set_index("Date").resample(freq_map[granularity])["Total Price"].sum().reset_index()

            fig = px.line(ts, x="Date", y="Total Price", markers=True)
            fig.update_layout(height=380, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-title">Units Sold Over Time</div>', unsafe_allow_html=True)
            ts_units = df.set_index("Date").resample(freq_map[granularity])["Quantity"].sum().reset_index()
            fig3 = px.area(ts_units, x="Date", y="Quantity")
            fig3.update_layout(height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig3, use_container_width=True)

    # ========================================================================
    # REGION-WISE
    # ========================================================================
    elif menu == "🌍 Region-wise":
        region_sales = df.groupby("Region").agg(
            Revenue=("Total Price", "sum"),
            Units=("Quantity", "sum"),
            Orders=("Order ID", "nunique")
        ).reset_index().sort_values("Revenue", ascending=False)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(region_sales, x="Region", y="Revenue", color="Revenue",
                         color_continuous_scale="Blues", text_auto=".2s")
            fig.update_layout(height=400, margin=dict(t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(region_sales, use_container_width=True, hide_index=True)

        st.download_button("⬇️ Download Region Data", to_csv_bytes(region_sales),
                            "region_sales.csv", "text/csv")

    # ========================================================================
    # CATEGORY
    # ========================================================================
    elif menu == "🏷️ Category":
        category_revenue = df.groupby("Category").agg(
            Revenue=("Total Price", "sum"),
            Units=("Quantity", "sum"),
            AvgUnitPrice=("Unit Price", "mean")
        ).reset_index().sort_values("Revenue", ascending=False)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.treemap(df, path=["Category", "Product Name"], values="Total Price",
                              color="Total Price", color_continuous_scale="Greens")
            fig.update_layout(height=420, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(category_revenue, use_container_width=True, hide_index=True)

        st.download_button("⬇️ Download Category Data", to_csv_bytes(category_revenue),
                            "category_revenue.csv", "text/csv")

    # ========================================================================
    # TOP PRODUCTS
    # ========================================================================
    elif menu == "🏆 Top Products":
        product_sales = df.groupby("Product Name")[["Quantity", "Total Price"]].sum().reset_index()

        colA, colB = st.columns(2)
        basis = colA.selectbox("Rank By", ["Total Price", "Quantity"])

        n_products = len(product_sales)
        if n_products <= 1:
            top_n = 1
            colB.caption("Only one product in the filtered data.")
        else:
            top_n = colB.slider("Top N Products", min_value=1, max_value=n_products,
                                 value=min(5, n_products))

        top_products = product_sales.sort_values(by=basis, ascending=False).head(top_n)

        fig = px.bar(top_products.sort_values(basis), x=basis, y="Product Name",
                     orientation="h", color=basis, color_continuous_scale="Oranges",
                     text_auto=".2s")
        fig.update_layout(height=max(300, top_n * 40), margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(top_products, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Top Products", to_csv_bytes(top_products),
                            "top_products.csv", "text/csv")

    # ========================================================================
    # PAYMENT METHODS
    # ========================================================================
    elif menu == "💳 Payment Methods":
        payment_summary = df.groupby("Payment Method").agg(
            Revenue=("Total Price", "sum"),
            Orders=("Order ID", "nunique"),
            AvgOrderValue=("Total Price", "mean")
        ).reset_index().sort_values("Revenue", ascending=False)

        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.pie(payment_summary, names="Payment Method", values="Revenue", hole=0.5)
            fig.update_layout(margin=dict(t=10, b=10), height=380)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = px.bar(payment_summary, x="Payment Method", y="Orders", color="Payment Method",
                          text_auto=True)
            fig2.update_layout(showlegend=False, margin=dict(t=10, b=10), height=380)
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(payment_summary, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Payment Data", to_csv_bytes(payment_summary),
                            "payment_methods.csv", "text/csv")

    # ========================================================================
    # FORECAST (ML)
    # ========================================================================
    elif menu == "🔮 Forecast":
        st.markdown('<div class="section-title">Revenue Forecast <span class="ml-badge">Gradient Boosting</span></div>',
                    unsafe_allow_html=True)

        if not has_date:
            st.info("Forecasting requires a valid 'Date' column.")
        else:
            colA, colB, colC = st.columns(3)
            freq_label = colA.selectbox("Frequency", ["Daily", "Weekly", "Monthly"], index=0)
            freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
            freq = freq_map[freq_label]

            horizon_default = {"D": 30, "W": 8, "ME": 6}[freq]
            horizon = colB.slider("Forecast Horizon", min_value=3, max_value=90, value=horizon_default)
            run_cv = colC.checkbox("Show model performance (cross-validation)", value=True)

            ts_features = build_forecasting_features(df, freq=freq)
            min_required_rows = 40

            if len(ts_features.dropna()) < min_required_rows:
                st.warning(
                    f"Not enough historical data after feature engineering "
                    f"({len(ts_features.dropna())} usable rows). Try a wider date range, "
                    f"fewer filters, or a coarser frequency."
                )
            else:
                with st.spinner("Training forecasting model..."):
                    forecaster = SalesForecaster()

                    if run_cv:
                        metrics = forecaster.evaluate(ts_features, n_splits=min(4, max(2, len(ts_features) // 60)))
                        m1, m2, m3 = st.columns(3)
                        kpi_card("MAE", f"₹ {metrics['MAE']:,.0f}", m1)
                        kpi_card("MAPE", f"{metrics['MAPE']:.1%}", m2)
                        kpi_card("R²", f"{metrics['R2']:.3f}", m3)
                        st.caption("Metrics from time-series cross-validation (out-of-sample, walk-forward).")

                    forecaster.fit(ts_features)
                    future = forecaster.forecast_future(ts_features, horizon=horizon)

                fig = px.line(title=None)
                fig.add_scatter(x=ts_features["Date"], y=ts_features["Revenue"], mode="lines", name="Historical")
                fig.add_scatter(x=future["Date"], y=future["Revenue"], mode="lines+markers",
                                name="Forecast", line=dict(dash="dash"))
                fig.update_layout(height=420, margin=dict(t=10, b=10), legend=dict(orientation="h", y=1.05))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown('<div class="section-title">Forecasted Values</div>', unsafe_allow_html=True)
                st.dataframe(future, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download Forecast", to_csv_bytes(future),
                                    "revenue_forecast.csv", "text/csv")

                with st.expander("📊 What drives this forecast? (Feature importance)"):
                    importances = forecaster.feature_importances().head(10)
                    fig_imp = px.bar(importances, x="importance", y="feature", orientation="h",
                                      color="importance", color_continuous_scale="Purples")
                    fig_imp.update_layout(height=350, margin=dict(t=10, b=10),
                                          coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_imp, use_container_width=True)

    # ========================================================================
    # SEGMENTATION (ML)
    # ========================================================================
    elif menu == "🧩 Segmentation":
        st.markdown('<div class="section-title">Product / Region Segmentation <span class="ml-badge">K-Means</span></div>',
                    unsafe_allow_html=True)

        colA, colB = st.columns(2)
        level = colA.selectbox("Segment By", ["Product", "Region"])
        auto_k = colB.checkbox("Auto-select number of segments (silhouette score)", value=True)

        k = None
        if not auto_k:
            k = st.slider("Number of Segments (K)", min_value=2, max_value=6, value=4)

        level_key = "product" if level == "Product" else "region"
        min_rows = 6 if level_key == "product" else 3

        entity_count = df["Product Name"].nunique() if level_key == "product" else df["Region"].nunique()
        if entity_count < min_rows:
            st.warning(f"Need at least {min_rows} distinct {level.lower()}s to run segmentation "
                       f"(found {entity_count}). Adjust filters.")
        else:
            with st.spinner("Running clustering..."):
                seg_model, seg_result = run_segmentation(df, level=level_key, k=k)

            st.caption(f"Selected K = **{seg_model.k}** segments")

            entity_col = "Product Name" if level_key == "product" else "Region"
            x_metric = "TotalRevenue"
            y_metric = "PurchaseFrequency" if level_key == "product" else "AvgOrderValue"

            fig = px.scatter(
                seg_result, x=x_metric, y=y_metric, color="Segment",
                hover_name=entity_col, size="TotalRevenue", size_max=40
            )
            fig.update_layout(height=450, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-title">Segment Summary</div>', unsafe_allow_html=True)
            summary = seg_result.groupby("Segment")[["TotalRevenue", "TotalUnits"]].mean().round(0).reset_index()
            st.dataframe(summary, use_container_width=True, hide_index=True)

            st.markdown('<div class="section-title">Full Segmentation Results</div>', unsafe_allow_html=True)
            display_cols = [entity_col, "TotalRevenue", "TotalUnits", "Segment"]
            st.dataframe(
                seg_result[display_cols].sort_values("TotalRevenue", ascending=False),
                use_container_width=True, hide_index=True
            )
            st.download_button("⬇️ Download Segmentation", to_csv_bytes(seg_result),
                                f"{level_key}_segmentation.csv", "text/csv")

    # ========================================================================
    # ANOMALY DETECTION (ML)
    # ========================================================================
    elif menu == "🚨 Anomaly Detection":
        st.markdown('<div class="section-title">Order Anomaly Detection <span class="ml-badge">Isolation Forest</span></div>',
                    unsafe_allow_html=True)

        contamination = st.slider(
            "Expected Anomaly Rate", min_value=0.01, max_value=0.10, value=0.02, step=0.01,
            help="Approximate proportion of orders expected to be anomalous"
        )

        if len(df) < 30:
            st.warning("Need at least 30 orders to run anomaly detection reliably. Adjust filters.")
        else:
            with st.spinner("Scoring orders..."):
                detector = AnomalyDetector(contamination=contamination)
                result = detector.fit_predict(df)

            n_anomalies = int(result["IsAnomaly"].sum())
            k1, k2 = st.columns(2)
            kpi_card("Anomalies Flagged", f"{n_anomalies:,}", k1)
            kpi_card("Anomaly Rate", f"{n_anomalies/len(df):.1%}", k2)

            fig = px.scatter(
                result, x="Quantity", y="Unit Price", color="IsAnomaly",
                color_discrete_map={True: "#dc2626", False: "#94a3b8"},
                hover_data=["Order ID", "Category", "Product Name", "Total Price"],
                opacity=0.7
            )
            fig.update_layout(height=420, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-title">Flagged Orders</div>', unsafe_allow_html=True)
            cols = ["Order ID", "Date", "Category", "Product Name", "Quantity",
                   "Unit Price", "Total Price", "Region", "AnomalyScore"]
            anomalies = result[result["IsAnomaly"]][cols].sort_values("AnomalyScore")
            st.dataframe(anomalies, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download Anomalies", to_csv_bytes(anomalies),
                                "anomalous_orders.csv", "text/csv")

    # ========================================================================
    # DEEP FILTER
    # ========================================================================
    elif menu == "🔍 Deep Filter":
        colX, colY, colZ = st.columns(3)

        filter_categories = colX.multiselect("Category", categories, default=categories)
        min_revenue = colY.number_input("Minimum Order Value", min_value=0, value=0)
        sort_by = colZ.selectbox("Sort By", ["Total Price", "Quantity", "Unit Price"])

        filtered_df = df[
            df["Category"].isin(filter_categories) &
            (df["Total Price"] >= min_revenue)
        ].sort_values(by=sort_by, ascending=False)

        st.caption(f"Showing **{len(filtered_df):,}** of {len(df):,} records")
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Filtered Data", to_csv_bytes(filtered_df),
                            "filtered_sales.csv", "text/csv")

    # ========================================================================
    # RAW DATA
    # ========================================================================
    elif menu == "📋 Raw Data":
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Current View", to_csv_bytes(df),
                            "sales_data.csv", "text/csv")

else:
    st.info("👈 Upload a CSV file from the sidebar, or check **'Use bundled sample dataset'** to explore the demo.")
    with st.expander("📋 Expected CSV Format"):
        st.code(
            "Order ID,Date,Category,Product Name,Quantity,Unit Price,Total Price,Region,Payment Method\n"
            "ORD1001,2024-01-15,Electronics,Wireless Mouse,5,799,3995,North America,Credit Card\n"
            "ORD1002,2024-01-16,Home Appliances,Office Chair,2,4500,9000,Europe,PayPal"
        )