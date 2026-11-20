# ============================================================
# European Household Energy — Streamlit Live Demo
# Single-file app: Home + Data + Prediction + Forecasting
#                 + Clustering + Model Report
#
# Run:  streamlit run app.py
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pathlib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ------------------------------------------------------------
# 0. PATHS & CONFIG
# ------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent
MODELS     = ROOT / "models"
PROCESSED  = ROOT / "data" / "processed"
OUTPUTS    = ROOT / "outputs"
REPORTS    = OUTPUTS / "evaluation_reports"
PRED_DIR   = OUTPUTS / "predictions"
FIGURES    = OUTPUTS / "figures"

PALETTE = {
    "primary":   "#1f4e79",
    "accent":    "#f2a900",
    "renewable": "#2ca02c",
    "fossil":    "#d62728",
    "neutral":   "#7f7f7f",
    "bg":        "#fbfbfb",
}

CUSTOM_CSS = f"""
<style>
    .stApp {{ background-color: {PALETTE['bg']}; }}
    .kpi-card {{
        background: white; padding: 18px 22px; border-radius: 12px;
        border-left: 5px solid {PALETTE['primary']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 12px;
    }}
    .kpi-label {{
        font-size: 13px; color: {PALETTE['neutral']};
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;
    }}
    .kpi-value {{ font-size: 26px; font-weight: 700; color: {PALETTE['primary']}; }}
    .insight-box {{
        background: #f0f6fc; border-left: 4px solid {PALETTE['primary']};
        padding: 14px 18px; border-radius: 6px; margin: 14px 0;
        font-size: 14px; line-height: 1.6;
    }}
    .warn-box {{
        background: #fff7e6; border-left: 4px solid {PALETTE['accent']};
        padding: 14px 18px; border-radius: 6px; margin: 14px 0;
    }}
    h1, h2, h3 {{ color: {PALETTE['primary']}; }}
</style>
"""

st.set_page_config(
    page_title="European Household Energy ML",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------
# 1. CACHED LOADERS
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_prediction_bundle():
    try:
        model_obj    = joblib.load(MODELS / "best_prediction_model.joblib")
        preprocessor = joblib.load(MODELS / "preprocessor.joblib")
        return {
            "model":        model_obj["model"],
            "name":         model_obj.get("name", "HistGB"),
            "preprocessor": preprocessor,
        }
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def load_clustering_bundle():
    try:
        return joblib.load(MODELS / "clustering_bundle.joblib")
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def load_csv(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


# ------------------------------------------------------------
# 2. UI HELPERS
# ------------------------------------------------------------
def kpi_card(label, value, delta=None):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{delta_html}</div>',
        unsafe_allow_html=True,
    )


def insight_box(text, kind="info"):
    cls = "insight-box" if kind == "info" else "warn-box"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def show_image_if_exists(relpath, caption=None, width=None):
    p = ROOT / relpath
    if p.exists():
        st.image(str(p), caption=caption, use_column_width=True)
    else:
        st.info(f"Figure not available: {relpath}")


# ------------------------------------------------------------
# 3. PAGE HEADER
# ------------------------------------------------------------
st.title("🏠 European Household Energy — ML Intelligence System")
st.caption(
    "Prediction · Forecasting · Clustering on 15 years of Eurostat household energy data "
    "(2010–2024). Interactive live demo."
)

# ------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# ------------------------------------------------------------
PAGES = [
    "🏠 Home",
    "📊 Data Overview",
    "🔮 Prediction",
    "📈 Forecasting",
    "🎯 Clustering",
    "📋 Model Report",
]
page = st.sidebar.radio("Navigate", PAGES, index=0)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Built with Streamlit · Python · scikit-learn · statsmodels. "
    "Data source: Eurostat (CC-BY)."
)

# ============================================================
# PAGE 1 — HOME
# ============================================================
if page == "🏠 Home":
    country_total = load_csv(PROCESSED / "country_total.csv")
    panel         = load_csv(PROCESSED / "panel_level.csv")
    forecasts     = load_csv(PRED_DIR / "forecasts_2025_2027.csv")

    n_countries = int(country_total.country.nunique()) if country_total is not None else 0
    n_years     = int(country_total.year.nunique())    if country_total is not None else 0
    n_panel     = len(panel)                            if panel is not None else 0
    n_forecast  = len(forecasts)                        if forecasts is not None else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Countries", n_countries)
    with c2: kpi_card("Years", n_years)
    with c3: kpi_card("Panel rows", f"{n_panel:,}")
    with c4: kpi_card("Forecasts", n_forecast)

    st.markdown("---")
    st.subheader("🗺️ What this demo shows")

    left, right = st.columns(2)
    with left:
        st.markdown("### 🔮 Prediction")
        st.write("Point prediction of household energy (TJ) for country × year × end-use × fuel.")
        st.markdown("### 📈 Forecasting")
        st.write("Per-country 2025–2027 forecasts using time-series models.")
    with right:
        st.markdown("### 🎯 Clustering")
        st.write("Segments countries by household energy profile (KMeans).")
        st.markdown("### 📋 Model Report")
        st.write("All evaluation tables and figures in one place.")

    insight_box(
        "<b>Dataset:</b> Eurostat household energy, 33,874 raw rows. "
        "Cleaned to 15,813 panel rows × 35 end-use/fuel combinations × 30 countries. "
        "Zero classification and fuel-hierarchy tagging prevent double counting."
    )

# ============================================================
# PAGE 2 — DATA OVERVIEW
# ============================================================
elif page == "📊 Data Overview":
    st.header("📊 Data Overview")

    panel         = load_csv(PROCESSED / "panel_level.csv")
    country_total = load_csv(PROCESSED / "country_total.csv")

    if panel is None:
        st.error("Panel dataset not found at data/processed/panel_level.csv")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Rows", f"{len(panel):,}")
    with c2: kpi_card("Countries", int(panel.country.nunique()))
    with c3: kpi_card("Fuels", int(panel.fuel_code.nunique()))
    with c4: kpi_card("End-uses", int(panel.end_use_short.nunique() if "end_use_short" in panel.columns
                                     else panel.end_use_code.nunique()))
    with c5: kpi_card("Years", f"{panel.year.min()}–{panel.year.max()}")

    # Observation count by year
    st.subheader("Observation count by year")
    yc = panel.groupby("year").size().reset_index(name="rows")
    fig = px.bar(yc, x="year", y="rows", color_discrete_sequence=[PALETTE["primary"]])
    fig.update_layout(height=360, plot_bgcolor="white",
                      xaxis_title="Year", yaxis_title="Observations")
    st.plotly_chart(fig, use_container_width=True)

    insight_box(
        "<b>Reading:</b> Observation counts are stable from 2014 onward. "
        "2010–2013 have occasional gaps in Eastern European series (documented in Phase 2)."
    )

    # Country coverage heatmap
    st.subheader("Country × Year coverage")
    cov = panel.groupby(["country", "year"]).size().reset_index(name="rows")
    fig = px.density_heatmap(cov, x="year", y="country", z="rows",
                             color_continuous_scale="Blues", height=700)
    fig.update_layout(xaxis_title="Year", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    # Target distribution
    st.subheader("Target distribution — energy_tj")
    pos = panel.loc[panel.energy_tj > 0, ["energy_tj"]].copy()
    pos["log1p"] = np.log1p(pos["energy_tj"])
    t1, t2 = st.tabs(["Raw (positive only)", "log1p-transformed"])
    with t1:
        fig = px.histogram(pos, x="energy_tj", nbins=80,
                           color_discrete_sequence=[PALETTE["primary"]])
        fig.update_layout(height=360, plot_bgcolor="white",
                          xaxis_title="energy_tj", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        fig = px.histogram(pos, x="log1p", nbins=80,
                           color_discrete_sequence=[PALETTE["renewable"]])
        fig.update_layout(height=360, plot_bgcolor="white",
                          xaxis_title="log1p(energy_tj)", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    insight_box(
        "<b>Why log1p:</b> Raw target has skewness ≈ 20. log1p reduces skew and stabilizes "
        "training for gradient boosting and linear models."
    )

    # Top-10 countries
    if country_total is not None:
        st.subheader("Top 10 countries — average household energy")
        top10 = (country_total.groupby("country").energy_tj.mean()
                 .sort_values(ascending=False).head(10).reset_index())
        fig = px.bar(top10, x="energy_tj", y="country", orientation="h",
                     color_discrete_sequence=[PALETTE["primary"]])
        fig.update_layout(height=430, plot_bgcolor="white",
                          xaxis_title="Mean TJ (2010–2024)", yaxis_title="",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 3 — PREDICTION
# ============================================================
elif page == "🔮 Prediction":
    st.header("🔮 Energy Consumption Prediction")
    st.write(
        "Predict household energy (TJ) for a given country × year × end-use × fuel. "
        "Backed by the best model from Phase 3 (HistGradientBoosting by default)."
    )

    bundle = load_prediction_bundle()
    if bundle is None:
        st.error(
            "Prediction artifacts not found. Run Phase 3 notebook to produce:\n"
            "- models/best_prediction_model.joblib\n"
            "- models/preprocessor.joblib"
        )
        st.stop()

    # KPIs from model comparison
    comparison = load_csv(REPORTS / "phase3_model_comparison.csv")
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Best model", bundle["name"])
    if comparison is not None:
        best = comparison.iloc[0]
        with c2: kpi_card("Val WAPE", f"{best.get('val', '—')}%")
        with c3: kpi_card("Test WAPE", f"{best.get('test', '—')}%")
    with c4: kpi_card("Features (after OHE)", "55")

    st.markdown("---")
    panel = load_csv(PROCESSED / "panel_level.csv")
    if panel is None:
        st.error("panel_level.csv not found.")
        st.stop()

    with st.form("prediction_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            country = st.selectbox("Country", sorted(panel.country.unique()), index=0)
        with col2:
            year = st.slider("Year", 2015, 2027, 2024)
        with col3:
            end_use = st.selectbox("End-use", sorted(panel.end_use_short.unique()), index=0)
        with col4:
            fuel = st.selectbox("Fuel (code)", sorted(panel.fuel_code.unique()), index=0)
        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        X_input = pd.DataFrame([{
            "country": country,
            "end_use_short": end_use,
            "fuel_code": fuel,
            "year": year,
        }])
        try:
            X_t   = bundle["preprocessor"].transform(X_input)
            y_log = float(bundle["model"].predict(X_t)[0])
            y_tj  = max(float(np.expm1(y_log)), 0.0)

            st.subheader("📈 Result")
            cA, cB, cC = st.columns(3)
            with cA: kpi_card("Predicted energy", f"{y_tj:,.0f} TJ")
            with cB: kpi_card("In log space", f"{y_log:.3f}")
            with cC: kpi_card("Type", "Point estimate")

            insight_box(
                f"Prediction for <b>{country}</b> in <b>{year}</b>, end-use = <b>{end_use}</b>, "
                f"fuel = <b>{fuel}</b>: <b>{y_tj:,.0f} TJ</b>. "
                "Interpret alongside the test WAPE above."
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("Configure the inputs above and click **Predict**.")

    st.markdown("---")
    st.subheader("📊 Model comparison (Phase 3)")
    if comparison is not None:
        st.dataframe(comparison, use_container_width=True)
    else:
        st.info("phase3_model_comparison.csv not found.")

    st.subheader("📉 Residuals (test set)")
    show_image_if_exists("outputs/figures/phase3_residuals.png")

# ============================================================
# PAGE 4 — FORECASTING
# ============================================================
elif page == "📈 Forecasting":
    st.header("📈 Household Energy Forecasting (2025–2027)")
    st.write(
        "Per-country 3-year forecasts using Naive, Drift, Linear, SES, Holt, ARIMA(1,1,1). "
        "Each country uses the best backtested model."
    )

    country_total = load_csv(PROCESSED / "country_total.csv")
    forecasts     = load_csv(PRED_DIR / "forecasts_2025_2027.csv")
    backtest      = load_csv(REPORTS / "phase4_backtest_results.csv")
    best_per      = load_csv(REPORTS / "phase4_best_model_per_country.csv")

    if country_total is None or forecasts is None:
        st.error("Forecasting artifacts not found. Run Phase 4 notebook.")
        st.stop()

    summary = (forecasts.groupby("year")
               .agg(total=("forecast_tj", "sum"),
                    mean_change=("change_vs_last_pct", "mean"))
               .reset_index())

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Countries", int(forecasts.country.nunique()))
    with c2: kpi_card("Horizon", "2025–2027")
    with c3: kpi_card("2027 total", f"{summary.total.iloc[-1]:,.0f} TJ")
    with c4: kpi_card("Mean 2027 Δ", f"{summary.mean_change.iloc[-1]:+.1f}%")

    st.markdown("---")
    st.subheader("🏁 Backtest performance — model comparison")
    if backtest is not None:
        st.dataframe(backtest, use_container_width=True)
    else:
        st.info("Backtest table not found.")

    st.subheader("🌍 Country-level forecast")
    country = st.selectbox("Choose a country", sorted(forecasts.country.unique()), index=0)

    hist = country_total[country_total.country == country].sort_values("year")
    fc   = forecasts[forecasts.country == country].sort_values("year")

    best_model_name = "—"
    if best_per is not None:
        row = best_per[best_per.country == country]
        if not row.empty:
            best_model_name = row.iloc[0]["best_model"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.year, y=hist.energy_tj,
                             mode="lines+markers", name="Historical",
                             line=dict(color=PALETTE["primary"], width=3)))
    fig.add_trace(go.Scatter(x=fc.year, y=fc.forecast_tj,
                             mode="lines+markers",
                             name=f"Forecast ({best_model_name})",
                             line=dict(color=PALETTE["fossil"], dash="dash", width=3)))
    fig.add_vline(x=2024.5, line_dash="dot", line_color="grey")
    fig.update_layout(height=500, plot_bgcolor="white",
                      xaxis_title="Year", yaxis_title="Energy (TJ)",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Forecast table")
    st.dataframe(
        fc[["year", "forecast_tj", "change_vs_last_pct", "model"]]
        .style.format({"forecast_tj": "{:,.0f}", "change_vs_last_pct": "{:+.2f}%"}),
        use_container_width=True,
    )

    st.subheader("🔝 Top 6 countries — forecast trajectory")
    top6 = (country_total.groupby("country").energy_tj.mean()
            .sort_values(ascending=False).head(6).index.tolist())
    fig = px.line(forecasts[forecasts.country.isin(top6)],
                  x="year", y="forecast_tj", color="country", markers=True)
    fig.update_layout(height=450, plot_bgcolor="white",
                      xaxis_title="Year", yaxis_title="Forecasted TJ")
    st.plotly_chart(fig, use_container_width=True)

    insight_box(
        "<b>Reading:</b> Each country uses its own best model — trend-followers vs mean-reverters. "
        "All projections assume historical patterns continue; no policy shocks are modelled."
    )

# ============================================================
# PAGE 5 — CLUSTERING
# ============================================================
elif page == "🎯 Clustering":
    st.header("🎯 Country Clusters — Unsupervised Segmentation")
    st.write(
        "KMeans segmentation of European countries by household energy profile "
        "(scale, volatility, growth, fuel mix, end-use mix)."
    )

    bundle   = load_clustering_bundle()
    clusters = load_csv(REPORTS / "phase5_clusters.csv")
    profile  = load_csv(REPORTS / "phase5_cluster_profile.csv")

    if bundle is None or clusters is None:
        st.warning(
            "Clustering artifacts not found. Run Phase 5 notebook to generate:\n"
            "- models/clustering_bundle.joblib\n"
            "- outputs/evaluation_reports/phase5_clusters.csv"
        )
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Countries", len(clusters))
    with c2: kpi_card("Clusters (k)", bundle.get("k", "—"))
    with c3: kpi_card("Features", len(bundle.get("features", [])))
    with c4: kpi_card("Algorithm", "KMeans")

    st.markdown("---")
    st.subheader("🗺️ PCA 2D projection")
    show_image_if_exists("outputs/figures/phase5_pca_clusters.png")

    st.subheader("📊 Cluster profile table")
    if profile is not None:
        st.dataframe(profile, use_container_width=True)

    st.subheader("🔍 Which cluster does a country belong to?")
    country = st.selectbox("Choose a country", sorted(clusters.country.unique()), index=0)
    row = clusters[clusters.country == country].iloc[0]
    st.markdown(
        f"**{country}** belongs to **Cluster {row['cluster_kmeans']}** — "
        f"*{row.get('cluster_label', 'Unlabeled')}*"
    )
    st.json({
        "avg_total_tj": float(row["avg_total_tj"]),
        "cv_total":     float(row["cv_total"]),
        "growth_pct":   float(row["growth_pct"]),
        "elec_share":   float(row["elec_share"]),
        "gas_share":    float(row["gas_share"]),
    })

    insight_box(
        "<b>Interpretation:</b> Clusters are descriptive, not normative. "
        "Use them to design targeted policies, not to rank countries."
    )

# ============================================================
# PAGE 6 — MODEL REPORT
# ============================================================
elif page == "📋 Model Report":
    st.header("📋 Model Report")
    st.write("Consolidated evaluation artifacts for all ML modules.")

    st.subheader("🔮 Phase 3 — Prediction")
    comp = load_csv(REPORTS / "phase3_model_comparison.csv")
    if comp is not None:
        st.dataframe(comp, use_container_width=True)
    imp = load_csv(REPORTS / "phase3_feature_importance.csv")
    if imp is not None:
        st.dataframe(imp, use_container_width=True)
    show_image_if_exists("outputs/figures/phase3_feature_importance.png")

    st.subheader("📈 Phase 4 — Forecasting")
    bt = load_csv(REPORTS / "phase4_backtest_results.csv")
    if bt is not None:
        st.dataframe(bt, use_container_width=True)
    bp = load_csv(REPORTS / "phase4_best_model_per_country.csv")
    if bp is not None:
        st.dataframe(bp, use_container_width=True)

    st.subheader("🎯 Phase 5 — Clustering")
    cprof = load_csv(REPORTS / "phase5_cluster_profile.csv")
    if cprof is not None:
        st.dataframe(cprof, use_container_width=True)

    insight_box(
        "<b>Consistency:</b> Prediction WAPE validates the panel model; "
        "Forecasting backtests validate country trajectories; Clustering provides segmentation."
    )

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption(
    "Educational demo · Eurostat data (CC-BY) · Built with Python & Streamlit."
)


