#===========================================================
# European Household Energy — Professional ML Dashboard
# Auto light/dark mode · Material icons · Polished UI
# Includes: Home · Data · Prediction · Forecasting · Clustering
#           · Model Report · Country Diagnostics
# Run: streamlit run app.py
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ------------------------------------------------------------
# 0. PATH RESOLUTION
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

PROCESSED_CANDIDATES = [
    ROOT / "data" / "processed",
    ROOT.parent / "data" / "processed",
    ROOT / "notebooks" / "data" / "processed",
    ROOT.parent / "European_Household_Energy_Project" / "data" / "processed",
    ROOT / "European_Household_Energy_Project" / "data" / "processed",
]

PROCESSED = next(
    (f for f in PROCESSED_CANDIDATES if (f / "panel_level.csv").exists()),
    None,
)
if PROCESSED is None:
    PROCESSED = next(
        (f for f in PROCESSED_CANDIDATES if (f / "country_total.csv").exists()),
        None,
    )

DATA_ROOT    = PROCESSED.parent if PROCESSED else (ROOT / "data")
PROJECT_ROOT = DATA_ROOT.parent if PROCESSED else ROOT

MODELS   = PROJECT_ROOT / "models"
OUTPUTS  = PROJECT_ROOT / "outputs"
REPORTS  = OUTPUTS / "evaluation_reports"
PRED_DIR = OUTPUTS / "predictions"
FIGURES  = OUTPUTS / "figures"

if not MODELS.exists():
    MODELS = ROOT / "models"
if not OUTPUTS.exists():
    OUTPUTS = ROOT / "outputs"
    REPORTS = OUTPUTS / "evaluation_reports"
    PRED_DIR = OUTPUTS / "predictions"
    FIGURES = OUTPUTS / "figures"

# ------------------------------------------------------------
# 1. THEME TOKENS (light + dark)
# ------------------------------------------------------------
LIGHT = {
    "bg":        "#f8fafc",
    "bg_soft":   "#f1f5f9",
    "surface":   "#ffffff",
    "border":    "#e2e8f0",
    "text":      "#0f172a",
    "text_mute": "#64748b",
    "primary":   "#4f46e5",
    "primary_2": "#6366f1",
    "accent":    "#10b981",
    "danger":    "#ef4444",
    "warning":   "#f59e0b",
    "shadow":    "rgba(15,23,42,0.06)",
    "shadow_2":  "rgba(15,23,42,0.10)",
    "chart_bg":  "#ffffff",
    "chart_fg":  "#0f172a",
    "chart_grid":"#e2e8f0",
}

DARK = {
    "bg":        "#0b1120",
    "bg_soft":   "#111827",
    "surface":   "#1e293b",
    "border":    "#334155",
    "text":      "#f1f5f9",
    "text_mute": "#94a3b8",
    "primary":   "#818cf8",
    "primary_2": "#a5b4fc",
    "accent":    "#34d399",
    "danger":    "#f87171",
    "warning":   "#fbbf24",
    "shadow":    "rgba(0,0,0,0.35)",
    "shadow_2":  "rgba(0,0,0,0.55)",
    "chart_bg":  "#1e293b",
    "chart_fg":  "#f1f5f9",
    "chart_grid":"#334155",
}

# ------------------------------------------------------------
# 2. APP CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="European Household Energy — ML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown(
        '<div style="padding:8px 0 16px 0;">'
        '<div style="font-size:20px;font-weight:800;letter-spacing:-0.02em;">'
        '⚡ Energy ML'
        '</div>'
        '<div style="font-size:11px;opacity:.7;letter-spacing:.08em;'
        'text-transform:uppercase;">European Households</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    theme_choice = st.radio(
        "Appearance",
        ["☀️ Light", "🌙 Dark", "🖥️ Auto"],
        index=2,
        horizontal=False,
    )

system_theme = "Light"
if theme_choice == "🌙 Dark":
    is_dark = True
elif theme_choice == "☀️ Light":
    is_dark = False
else:
    is_dark = False

T = DARK if is_dark else LIGHT

# ------------------------------------------------------------
# 3. CSS
# ------------------------------------------------------------
CUSTOM_CSS = f"""
<style>
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: {T['text']};
    }}
    .stApp {{ background: {T['bg']}; }}
    h1, h2, h3, h4 {{ color: {T['text']}; letter-spacing: -0.02em; font-weight: 700; }}
    p, span, label, div {{ color: {T['text']}; }}

    .kpi {{
        background: {T['surface']};
        border: 1px solid {T['border']};
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 1px 2px {T['shadow']}, 0 4px 16px {T['shadow']};
        position: relative;
        transition: all .2s cubic-bezier(.4,0,.2,1);
        height: 100%;
    }}
    .kpi:hover {{
        border-color: {T['primary']}40;
        box-shadow: 0 4px 12px {T['shadow_2']}, 0 8px 32px {T['shadow']};
        transform: translateY(-2px);
    }}
    .kpi-icon {{
        width: 36px; height: 36px;
        border-radius: 10px;
        background: {T['primary']}14;
        color: {T['primary']};
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 14px;
        font-size: 18px;
        font-weight: 700;
    }}
    .kpi-label {{
        font-size: 11px;
        color: {T['text_mute']};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 26px;
        font-weight: 800;
        color: {T['text']};
        line-height: 1.15;
        letter-spacing: -0.02em;
    }}
    .kpi-delta {{
        font-size: 12px;
        margin-top: 8px;
        font-weight: 600;
        color: {T['accent']};
    }}

    .insight {{
        background: {T['primary']}08;
        border: 1px solid {T['primary']}25;
        border-left: 4px solid {T['primary']};
        padding: 16px 20px;
        border-radius: 10px;
        margin: 16px 0;
        font-size: 14px;
        line-height: 1.7;
        color: {T['text']};
    }}
    .warn {{
        background: {T['warning']}10;
        border: 1px solid {T['warning']}30;
        border-left: 4px solid {T['warning']};
        padding: 16px 20px;
        border-radius: 10px;
        margin: 16px 0;
        color: {T['text']};
    }}
    .success {{
        background: {T['accent']}10;
        border: 1px solid {T['accent']}30;
        border-left: 4px solid {T['accent']};
        padding: 16px 20px;
        border-radius: 10px;
        margin: 16px 0;
        color: {T['text']};
    }}
    .danger {{
        background: {T['danger']}10;
        border: 1px solid {T['danger']}30;
        border-left: 4px solid {T['danger']};
        padding: 16px 20px;
        border-radius: 10px;
        margin: 16px 0;
        color: {T['text']};
    }}

    section[data-testid="stSidebar"] {{
        background: {T['surface']} !important;
        border-right: 1px solid {T['border']};
    }}
    section[data-testid="stSidebar"] * {{ color: {T['text']} !important; }}
    section[data-testid="stSidebar"] .stRadio > div {{ gap: 4px; }}
    section[data-testid="stSidebar"] .stRadio label {{
        padding: 9px 14px !important;
        border-radius: 10px !important;
        cursor: pointer;
        transition: all .15s ease;
        font-size: 14px;
        font-weight: 500;
        color: {T['text_mute']} !important;
    }}
    section[data-testid="stSidebar"] .stRadio label:hover {{
        background: {T['primary']}0d;
        color: {T['text']} !important;
    }}
    section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
    section[data-testid="stSidebar"] .stRadio input:checked + div {{
        background: {T['primary']}14;
        color: {T['primary']} !important;
        font-weight: 600;
    }}

    .stButton > button, .stFormSubmitButton > button {{
        background: {T['primary']};
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 14px;
        transition: all .15s ease;
        box-shadow: 0 1px 2px {T['shadow']};
    }}
    .stButton > button:hover, .stFormSubmitButton > button:hover {{
        background: {T['primary_2']};
        box-shadow: 0 4px 12px {T['primary']}40;
        transform: translateY(-1px);
    }}

    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {{
        background: {T['surface']} !important;
        color: {T['text']} !important;
        border: 1px solid {T['border']} !important;
        border-radius: 10px !important;
    }}
    .stSlider > div > div > div {{ background: {T['primary']} !important; }}

    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid {T['border']};
        background: {T['surface']};
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        border-bottom: 1px solid {T['border']};
        background: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        color: {T['text_mute']};
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        color: {T['primary']} !important;
        background: {T['primary']}08 !important;
        border-bottom: 2px solid {T['primary']} !important;
    }}

    .js-plotly-plot .plotly .main-svg {{ background: transparent !important; }}
    hr {{ border-color: {T['border']} !important; }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: {T['text_mute']} !important; }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------
# 4. SVG ICONS
# ------------------------------------------------------------
def icon(name: str, size: int = 20, color: str = None) -> str:
    c = color or T["primary"]
    icons = {
        "globe":      '<path d="M12 2a10 10 0 100 20 10 10 0 000-20zm0 2a8 8 0 017.93 7H15.9a12 12 0 00-2.9-5.66A8 8 0 0112 4zm-2.86.34A12 12 0 006.24 9H4.07a8 8 0 014.99-4.66zM4.07 15h2.17a12 12 0 002.9 5.66A8 8 0 014.07 15zm4.17 0h7.52a10 10 0 01-3.76 5.4A10 10 0 018.24 15zm7.52-2H8.24a10 10 0 013.76-5.4A10 10 0 0115.76 13zm1.17 2h2.17a8 8 0 01-4.99 4.66A12 12 0 0016.93 15zm2.17-2h-2.17a12 12 0 00-2.9 5.66A8 8 0 0019.1 13z"/>',
        "calendar":   '<path d="M19 3h-1V1h-2v2H8V1H6v2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V8h14v11z"/>',
        "database":   '<path d="M12 3C7.58 3 4 4.79 4 7v10c0 2.21 3.58 4 8 4s8-1.79 8-4V7c0-2.21-3.58-4-8-4zm0 2c3.87 0 6 1.5 6 2s-2.13 2-6 2-6-1.5-6-2 2.13-2 6-2zm6 12c0 .5-2.13 2-6 2s-6-1.5-6-2v-2.23c1.61.78 3.72 1.23 6 1.23s4.39-.45 6-1.23V17zm0-5c0 .5-2.13 2-6 2s-6-1.5-6-2V9.77C7.61 10.55 9.72 11 12 11s4.39-.45 6-1.23V12z"/>',
        "chart":      '<path d="M3 13h2v8H3v-8zm4-4h2v12H7V9zm4-6h2v18h-2V3zm4 8h2v10h-2V11zm4-4h2v14h-2V7z"/>',
        "bolt":       '<path d="M11 21h-1l1-7H7.5c-.88 0-.33-.75-.31-.78C8.48 10.94 10.42 7.54 13.01 3h1l-1 7h3.51c.4 0 .62.19.4.66C12.97 17.55 11 21 11 21z"/>',
        "target":     '<path d="M12 2a10 10 0 100 20 10 10 0 000-20zm0 18a8 8 0 110-16 8 8 0 010 16zm0-13a5 5 0 100 10 5 5 0 000-10zm0 8a3 3 0 110-6 3 3 0 010 6z"/>',
        "cog":        '<path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>',
        "sparkles":   '<path d="M19 9l1.25-2.75L23 5l-2.75-1.25L19 1l-1.25 2.75L15 5l2.75 1.25L19 9zm-7.5.5L9 4 6.5 9.5 1 12l5.5 2.5L9 20l2.5-5.5L17 12l-5.5-2.5zM19 15l-1.25 2.75L15 19l2.75 1.25L19 23l1.25-2.75L23 19l-2.75-1.25L19 15z"/>',
        "report":     '<path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6zm2 18H6V4h7v5h5v11zM8 12h8v2H8zm0 4h8v2H8zm0-8h3v2H8z"/>',
        "trend":      '<path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/>',
        "grid":       '<path d="M3 3h8v8H3V3zm10 0h8v8h-8V3zM3 13h8v8H3v-8zm10 0h8v8h-8v-8z"/>',
        "stethoscope": '<path d="M19 8h-2V3h-2v5h-2V3h-2v6c0 1.66 1.34 3 3 3h1v3.5a3.5 3.5 0 01-7 0V9c0-1.1-.9-2-2-2s-2 .9-2 2v6.5C4 19.54 6.46 22 9.5 22s5.5-2.46 5.5-5.5V12h4v-2z"/>',
        "alert":      '<path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>',
        "check":      '<path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>',
    }
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'width="{size}" height="{size}" fill="{c}">{icons.get(name, icons["chart"])}</svg>'
    )


# ------------------------------------------------------------
# 5. LOADERS (Updated for Advanced Model)
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_prediction_bundle():
    # Try Advanced Model first
    try:
        bundle = joblib.load(MODELS / "advanced_prediction_bundle.joblib")
        return {
            "type": "advanced",
            "bundle": bundle,
            "name": bundle.get("model_name", "Advanced"),
        }
    except FileNotFoundError:
        pass
    
    # Fallback to Legacy Model
    try:
        model_obj = joblib.load(MODELS / "best_prediction_model.joblib")
        preprocessor = joblib.load(MODELS / "preprocessor.joblib")
        return {
            "type": "legacy",
            "model": model_obj["model"],
            "name": model_obj.get("name", "HistGB"),
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
    if path and path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data(show_spinner=False)
def load_advanced_forecasts(path: Path):
    """Load country forecasts exported by the Advanced Forecasting notebook."""
    if path and path.exists():
        df = pd.read_csv(path)
        required = {"country", "year", "forecast_tj"}
        if required.issubset(df.columns):
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df["forecast_tj"] = pd.to_numeric(df["forecast_tj"], errors="coerce")
            df = df.dropna(subset=["country", "year", "forecast_tj"])
            df["year"] = df["year"].astype(int)
            df = df[df["year"].between(2025, 2029)].copy()
            return df.sort_values(["country", "year"])
    return None


def load_forecast_source():
    # Prefer the five-year output from Advanced Forecasting; otherwise legacy Phase 4/6.
    advanced = load_advanced_forecasts(OUTPUTS / "advanced_forecasting" / "country_forecasts_2025_2029.csv")
    if advanced is not None and not advanced.empty:
        return advanced, "Advanced Forecasting (2025–2029)"
    legacy = load_csv(PRED_DIR / "forecasts_2025_2027_with_ci.csv")
    if legacy is None:
        legacy = load_csv(PRED_DIR / "forecasts_2025_2027.csv")
    return legacy, "Phase 4/6 legacy forecasts"


# ------------------------------------------------------------
# 6. UI HELPERS
# ------------------------------------------------------------
def kpi(label, value, icon_name="chart", delta=None):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="kpi">'
        f'<div class="kpi-icon">{icon(icon_name, 18)}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'{delta_html}</div>',
        unsafe_allow_html=True,
    )


def info_box(text, kind="info"):
    cls = {"info": "insight", "warn": "warn", "success": "success", "danger": "danger"}[kind]
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def show_figure(relpath, caption=None):
    p_fig = FIGURES / Path(relpath).name
    p_root = ROOT / relpath
    if p_fig.exists():
        st.image(str(p_fig), caption=caption, use_container_width=True)
    elif p_root.exists():
        st.image(str(p_root), caption=caption, use_container_width=True)
    else:
        st.caption(f"Figure not available: {relpath}")


def styled_plotly(fig):
    fig.update_layout(
        paper_bgcolor=T["chart_bg"],
        plot_bgcolor=T["chart_bg"],
        font=dict(color=T["chart_fg"], family="Inter, sans-serif", size=13),
        title_font=dict(color=T["chart_fg"], size=16, family="Inter, sans-serif"),
        xaxis=dict(gridcolor=T["chart_grid"], linecolor=T["chart_grid"],
                   zerolinecolor=T["chart_grid"], tickfont=dict(color=T["chart_fg"])),
        yaxis=dict(gridcolor=T["chart_grid"], linecolor=T["chart_grid"],
                   zerolinecolor=T["chart_grid"], tickfont=dict(color=T["chart_fg"])),
        legend=dict(font=dict(color=T["chart_fg"]), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


# ------------------------------------------------------------
# 7. HEADER
# ------------------------------------------------------------
st.markdown(
    f"""
    <div style="margin-bottom:24px;">
        <div style="display:flex;align-items:center;gap:14px;">
            <div style="width:44px;height:44px;border-radius:12px;
                        background:{T['primary']}14;color:{T['primary']};
                        display:flex;align-items:center;justify-content:center;">
                {icon("bolt", 22)}
            </div>
            <div>
                <div style="font-size:26px;font-weight:800;
                            letter-spacing:-0.02em;color:{T['text']};">
                    Household Energy Intelligence
                </div>
                <div style="font-size:13px;color:{T['text_mute']};margin-top:2px;">
                    European countries · 2010–2024 · ML dashboard
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("---")
    if PROCESSED is not None:
        st.markdown(
            f'<div style="font-size:11px;color:{T["text_mute"]};'
            f'text-transform:uppercase;letter-spacing:.08em;'
            f'margin-bottom:6px;">Data source</div>'
            f'<code style="font-size:11px;color:{T["text"]};'
            f'background:{T["bg_soft"]};padding:4px 8px;border-radius:6px;'
            f'display:block;word-break:break-all;">{PROCESSED}</code>',
            unsafe_allow_html=True,
        )
    else:
        st.error("Processed data not found")

# ------------------------------------------------------------
# 8. NAVIGATION
# ------------------------------------------------------------
PAGES = [
    "🏠 Home",
    "📊 Data Overview",
    "🔮 Prediction",
    "📈 Forecasting",
    "🩺 Country Diagnostics",
    "🎯 Clustering",
    "📋 Model Report",
]
page = st.sidebar.radio("", PAGES, index=0, label_visibility="collapsed")


# ============================================================
# PAGE — HOME
# ============================================================
if page == "🏠 Home":
    country_total = load_csv(PROCESSED / "country_total.csv") if PROCESSED else None
    panel         = load_csv(PROCESSED / "panel_level.csv") if PROCESSED else None
    forecasts, forecast_source = load_forecast_source()

    n_countries = int(country_total.country.nunique()) if country_total is not None else 0
    n_years     = int(country_total.year.nunique()) if country_total is not None else 0
    n_panel     = len(panel) if panel is not None else 0
    n_forecast  = len(forecasts) if forecasts is not None else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Countries", n_countries, "globe")
    with c2: kpi("Years covered", n_years, "calendar")
    with c3: kpi("Panel observations", f"{n_panel:,}", "database")
    with c4: kpi("Forecasts generated", n_forecast, "sparkles")

    st.markdown("---")
    st.subheader("What this dashboard shows")

    left, right = st.columns(2)
    with left:
        st.markdown(
            f'<div class="kpi" style="height:auto;">'
            f'<div class="kpi-icon">{icon("bolt", 18)}</div>'
            f'<div style="font-weight:700;font-size:15px;margin-bottom:6px;">Prediction</div>'
            f'<div style="font-size:13px;color:{T["text_mute"]};">'
            f'Point prediction of household energy for a given country × year × end-use × fuel.</div>'
            f'</div>', unsafe_allow_html=True,
        )
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="kpi" style="height:auto;">'
            f'<div class="kpi-icon">{icon("trend", 18)}</div>'
            f'<div style="font-weight:700;font-size:15px;margin-bottom:6px;">Forecasting</div>'
            f'<div style="font-size:13px;color:{T["text_mute"]};">'
            f'Per-country 2025–2027 forecasts with confidence intervals.</div>'
            f'</div>', unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f'<div class="kpi" style="height:auto;">'
            f'<div class="kpi-icon">{icon("stethoscope", 18)}</div>'
            f'<div style="font-weight:700;font-size:15px;margin-bottom:6px;">Country Diagnostics</div>'
            f'<div style="font-size:13px;color:{T["text_mute"]};">'
            f'Pick a country → see its energy challenges + comparison with Europe.</div>'
            f'</div>', unsafe_allow_html=True,
        )
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="kpi" style="height:auto;">'
            f'<div class="kpi-icon">{icon("target", 18)}</div>'
            f'<div style="font-weight:700;font-size:15px;margin-bottom:6px;">Clustering</div>'
            f'<div style="font-size:13px;color:{T["text_mute"]};">'
            f'KMeans segmentation of countries by household energy profile.</div>'
            f'</div>', unsafe_allow_html=True,
        )

    if PROCESSED is None:
        info_box(
            "<b>Data not found.</b> Run <b>Phase 2</b> notebook to generate "
            "<code>data/processed/panel_level.csv</code>.", kind="warn",
        )
    else:
        info_box(
            "<b>Dataset:</b> Eurostat household energy, 33,874 raw rows. Cleaned to "
            "15,813 panel rows × 35 end-use/fuel combinations × 30 countries."
        )


# ============================================================
# PAGE — DATA OVERVIEW
# ============================================================
elif page == "📊 Data Overview":
    st.header("Data Overview")

    if PROCESSED is None:
        st.error("Panel dataset not found. Run Phase 2 notebook first.")
        st.stop()

    panel = load_csv(PROCESSED / "panel_level.csv")
    country_total = load_csv(PROCESSED / "country_total.csv")

    if panel is None:
        st.error(f"panel_level.csv not found inside {PROCESSED}.")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Rows", f"{len(panel):,}", "database")
    with c2: kpi("Countries", int(panel.country.nunique()), "globe")
    with c3: kpi("Fuels", int(panel.fuel_code.nunique()), "grid")
    with c4: kpi("End-uses", int(panel.end_use_short.nunique() if "end_use_short" in panel.columns else panel.end_use_code.nunique()), "target")
    with c5: kpi("Years", f"{panel.year.min()}–{panel.year.max()}", "calendar")

    st.subheader("Observation count by year")
    yc = panel.groupby("year").size().reset_index(name="rows")
    fig = px.bar(yc, x="year", y="rows", color_discrete_sequence=[T["primary"]])
    fig.update_layout(height=360, xaxis_title="Year", yaxis_title="Observations")
    st.plotly_chart(styled_plotly(fig), use_container_width=True)

    st.subheader("Country × Year coverage")
    cov = panel.groupby(["country", "year"]).size().reset_index(name="rows")
    fig = px.density_heatmap(cov, x="year", y="country", z="rows",
                             color_continuous_scale=[[0, T["bg_soft"]], [1, T["primary"]]],
                             height=700)
    fig.update_layout(xaxis_title="Year", yaxis_title="")
    st.plotly_chart(styled_plotly(fig), use_container_width=True)

    st.subheader("Target distribution")
    pos = panel.loc[panel.energy_tj > 0, ["energy_tj"]].copy()
    pos["log1p"] = np.log1p(pos["energy_tj"])
    t1, t2 = st.tabs(["Raw", "log1p"])
    with t1:
        fig = px.histogram(pos, x="energy_tj", nbins=80, color_discrete_sequence=[T["primary"]])
        fig.update_layout(height=360, xaxis_title="energy_tj", yaxis_title="Frequency")
        st.plotly_chart(styled_plotly(fig), use_container_width=True)
    with t2:
        fig = px.histogram(pos, x="log1p", nbins=80, color_discrete_sequence=[T["accent"]])
        fig.update_layout(height=360, xaxis_title="log1p(energy_tj)", yaxis_title="Frequency")
        st.plotly_chart(styled_plotly(fig), use_container_width=True)

    if country_total is not None:
        st.subheader("Top 10 countries")
        top10 = (country_total.groupby("country").energy_tj.mean()
                 .sort_values(ascending=False).head(10).reset_index())
        fig = px.bar(top10, x="energy_tj", y="country", orientation="h",
                     color_discrete_sequence=[T["primary"]])
        fig.update_layout(height=430, xaxis_title="Mean TJ (2010–2024)", yaxis_title="",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(styled_plotly(fig), use_container_width=True)


# ============================================================
# PAGE — PREDICTION (Updated for Advanced Model)
# ============================================================
# ============================================================
# PAGE — PREDICTION (Updated for Advanced Model & Missing Data)
# ============================================================
elif page == "🔮 Prediction":
    st.header("Energy Consumption Prediction")
    st.caption("Predict household energy for a country × year × end-use × fuel.")

    bundle = load_prediction_bundle()
    panel = load_csv(PROCESSED / "panel_level.csv") if PROCESSED else None
    comparison = load_csv(REPORTS / "phase3_model_comparison.csv")

    if bundle is None:
        info_box(
            "<b>Model not found.</b> Expected "
            "<code>models/advanced_prediction_bundle.joblib</code> or "
            "<code>models/best_prediction_model.joblib</code>. "
            "Run the prediction notebook first.", kind="warn",
        )
    if panel is None:
        info_box(
            "<b>Panel data not found.</b> Expected "
            "<code>data/processed/panel_level.csv</code>. "
            "Run <b>Phase 2</b> notebook first.", kind="warn",
        )
    if bundle is None or panel is None:
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Active Model", bundle["name"], "bolt")
    if comparison is not None:
        best = comparison.iloc[0]
        with c2: kpi("Val WAPE", f"{best.get('val', '—')}%", "chart")
        with c3: kpi("Test WAPE", f"{best.get('test', '—')}%", "target")
    with c4: kpi("Model Type", bundle["type"].capitalize(), "cog")

    st.markdown("---")
    st.subheader("Configure a prediction")

    countries = sorted(panel.country.dropna().unique())
    end_uses = sorted(panel.end_use_short.dropna().unique()) if "end_use_short" in panel.columns else []
    fuels = sorted(panel.fuel_code.dropna().unique())

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            country = st.selectbox("Country", countries, index=0)
            end_use = st.selectbox("End-use category", end_uses, index=0)
        with col2:
            fuel = st.selectbox("Fuel type", fuels, index=0)
            year = st.slider("Target year", 2015, 2027, 2024)
        submitted = st.form_submit_button("Generate Prediction", use_container_width=True)

    if submitted:
        try:
            # 1. Try to find the exact historical match
            hist = panel[
                (panel.country == country) &
                (panel.end_use_short == end_use) &
                (panel.fuel_code == fuel)
            ].sort_values("year")

            # 2. Fallback logic if no exact history exists
            if hist.empty:
                st.warning(f"⚠️ No exact historical data found for **{country}** × **{end_use}** × **{fuel}**. Using smart fallback baseline...")
                
                # Fallback A: Country + Fuel (ignore end-use)
                hist_fallback_a = panel[(panel.country == country) & (panel.fuel_code == fuel)].sort_values("year")
                
                # Fallback B: Country + End-use (ignore fuel)
                hist_fallback_b = panel[(panel.country == country) & (panel.end_use_short == end_use)].sort_values("year")

                # Fallback C: Just Country overall average
                hist_fallback_c = panel[panel.country == country].sort_values("year")

                if not hist_fallback_a.empty:
                    hist = hist_fallback_a
                elif not hist_fallback_b.empty:
                    hist = hist_fallback_b
                elif not hist_fallback_c.empty:
                    hist = hist_fallback_c
                else:
                    st.error(f"Completely missing data for {country}.")
                    st.stop()

            hist_before = hist[hist.year < year]

            # 3. Calculate lags robustly
            if len(hist_before) >= 3:
                lag_1 = float(hist_before.iloc[-1]["energy_tj"])
                lag_2 = float(hist_before.iloc[-2]["energy_tj"])
                lag_3 = float(hist_before.iloc[-3]["energy_tj"])
            elif len(hist_before) == 2:
                lag_1 = float(hist_before.iloc[-1]["energy_tj"])
                lag_2 = float(hist_before.iloc[-2]["energy_tj"])
                lag_3 = lag_2
            elif len(hist_before) == 1:
                lag_1 = float(hist_before.iloc[-1]["energy_tj"])
                lag_2 = lag_1
                lag_3 = lag_1
            else:
                lag_1 = float(hist.iloc[0]["energy_tj"])
                lag_2 = lag_1
                lag_3 = lag_1

            rolling_mean = (lag_1 + lag_2 + lag_3) / 3
            rolling_std = float(np.std([lag_1, lag_2, lag_3])) if len(hist_before) >= 2 else 0.0

            y_tj = 0.0

            # 4. Execute prediction based on model type
            if bundle["type"] == "advanced":
                b = bundle["bundle"]
                end_use_code_val = panel[panel.end_use_short == end_use]["end_use_code"].iloc[0]
                year_index = int(year) - b["min_year"]
                
                X_input = pd.DataFrame([{
                    "country": country,
                    "end_use_code": end_use_code_val,
                    "fuel_code": fuel,
                    "year": int(year),
                    "year_index": year_index,
                    "year_sq": year_index ** 2,
                    "lag_1": lag_1,
                    "lag_2": lag_2,
                    "lag_3": lag_3,
                    "rolling_mean_3": rolling_mean,
                    "rolling_std_3": rolling_std
                }])
                y_tj = max(0.0, float(b["model"].predict(X_input[b["features"]])[0]))

            else: # legacy fallback
                X_input = pd.DataFrame([{
                    "country": country,
                    "end_use_short": end_use,
                    "fuel_code": fuel,
                    "year": year,
                    "prev_year_energy": lag_1,
                    "prev_2yr_energy": lag_2,
                    "rolling_2_prev": rolling_mean
                }])
                X_t = bundle["preprocessor"].transform(X_input)
                if hasattr(X_t, "toarray"):
                    X_t = X_t.toarray()
                y_log = float(bundle["model"].predict(np.asarray(X_t))[0])
                y_tj = max(float(np.expm1(y_log)), 0.0)

            st.markdown("---")
            st.subheader("Prediction result")

            cA, cB, cC = st.columns(3)
            with cA: kpi("Predicted energy", f"{y_tj:,.0f} TJ", "bolt")
            with cB: kpi("Model Used", bundle["name"], "cog")
            with cC: kpi("Confidence", "Point estimate", "target")

            info_box(
                f"<b>{country}</b> · <b>{year}</b> · <b>{end_use}</b> · <b>{fuel}</b><br>"
                f"Predicted household energy: "
                f"<b style='color:{T['primary']};font-size:18px;'>{y_tj:,.0f} TJ</b>"
            )

            st.markdown("#### Historical context + prediction")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.year, y=hist.energy_tj,
                mode="lines+markers", name="Historical (Fallback)" if len(hist) < len(panel) else "Historical",
                line=dict(color=T["primary"], width=2.5), marker=dict(size=8),
            ))
            fig.add_trace(go.Scatter(
                x=[year], y=[y_tj], mode="markers", name="Prediction",
                marker=dict(size=16, color=T["accent"], symbol="star",
                            line=dict(color=T["text"], width=1)),
            ))
            fig.add_vline(x=float(hist.year.max()) + 0.5, line_dash="dot", line_color=T["text_mute"])
            fig.update_layout(height=400, xaxis_title="Year", yaxis_title="Energy (TJ)",
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(styled_plotly(fig), use_container_width=True)

            last_val = float(hist_before.iloc[-1]["energy_tj"]) if len(hist_before) > 0 else y_tj
            delta_pct = (y_tj - last_val) / last_val * 100 if last_val > 0 else 0
            info_box(
                f"Compared to last observed (<b>{last_val:,.0f} TJ</b>): "
                f"<b style='color:{T['accent'] if delta_pct >= 0 else T['danger']};'>"
                f"{delta_pct:+.1f}%</b>"
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            with st.expander("Debug info"):
                st.write(f"Model Type: {bundle['type']}")
                st.write(f"Model Name: {bundle['name']}")

    st.markdown("---")
    st.subheader("Model comparison (Phase 3)")
    if comparison is not None:
        st.dataframe(comparison, use_container_width=True)

    st.subheader("Residuals (test set)")
    show_figure("outputs/figures/phase3_feature_importance.png") # fallback if residuals missing

# ============================================================
# PAGE — FORECASTING
# ============================================================
elif page == "📈 Forecasting":
    st.header("Household Energy Forecasting (2025–2029)")
    st.caption("Advanced country forecasts for 2025–2029. Confidence intervals are displayed when available.")

    country_total = load_csv(PROCESSED / "country_total.csv") if PROCESSED else None
    forecasts, forecast_source = load_forecast_source()
    if forecasts is not None and "lo80_tj" not in forecasts.columns:
        st.caption(f"Forecast source: {forecast_source} · Confidence intervals are not included in this export.")

    model_comp = load_csv(REPORTS / "phase6_model_comparison.csv")
    model_h = load_csv(REPORTS / "phase6_model_by_horizon.csv")
    best_per = load_csv(REPORTS / "phase4_best_model_per_country.csv")
    backtest = load_csv(REPORTS / "phase4_backtest_results.csv")

    if country_total is None or forecasts is None:
        info_box(
            "<b>Forecasting artifacts not found.</b> "
            "Run <b>Phase 4</b> or <b>Phase 6</b> notebook first.", kind="warn",
        )
        st.stop()

    has_ci = all(c in forecasts.columns for c in ["lo80_tj", "hi80_tj", "lo95_tj", "hi95_tj"])

    summary = forecasts.groupby("year").agg(total=("forecast_tj", "sum")).reset_index()
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Countries", int(forecasts.country.nunique()), "globe")
    with c2: kpi("Horizon", f"{int(forecasts.year.min())}–{int(forecasts.year.max())}", "calendar")
    with c3: kpi(f"{int(summary.year.iloc[-1])} total", f"{summary.total.iloc[-1]:,.0f} TJ", "bolt")
    with c4: kpi("CI available", "Yes" if has_ci else "No", "target")

    if model_comp is not None:
        st.subheader("Model comparison (Phase 6, if available)")
        st.dataframe(model_comp, use_container_width=True)

    if model_h is not None:
        st.subheader("WAPE by horizon step (Phase 6, if available)")
        st.dataframe(model_h, use_container_width=True)

    st.subheader("Country forecast")
    country = st.selectbox("Select a country", sorted(forecasts.country.unique()), index=0)

    hist = country_total[country_total.country == country].sort_values("year")
    fc = forecasts[forecasts.country == country].sort_values("year")

    best_model_name = forecast_source
    if best_per is not None:
        row = best_per[best_per.country == country]
        if not row.empty:
            best_model_name = row.iloc[0]["best_model"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.year, y=hist.energy_tj, mode="lines+markers",
                             name="Historical",
                             line=dict(color=T["primary"], width=3),
                             marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=fc.year, y=fc.forecast_tj, mode="lines+markers",
                             name=f"Forecast ({best_model_name})",
                             line=dict(color=T["accent"], dash="dash", width=3),
                             marker=dict(size=10, symbol="diamond")))

    if has_ci:
        fig.add_trace(go.Scatter(
            x=pd.concat([fc.year, fc.year[::-1]]),
            y=pd.concat([fc.hi95_tj, fc.lo95_tj[::-1]]),
            fill="toself", fillcolor="rgba(245,158,11,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([fc.year, fc.year[::-1]]),
            y=pd.concat([fc.hi80_tj, fc.lo80_tj[::-1]]),
            fill="toself", fillcolor="rgba(245,158,11,0.30)",
            line=dict(color="rgba(255,255,255,0)"),
            name="80% CI", showlegend=True,
        ))

    fig.add_vline(x=float(hist.year.max()) + 0.5, line_dash="dot", line_color=T["text_mute"])
    fig.update_layout(height=500, xaxis_title="Year", yaxis_title="Energy (TJ)",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(styled_plotly(fig), use_container_width=True)

    if has_ci:
        st.dataframe(
            fc[["year", "forecast_tj", "lo80_tj", "hi80_tj", "lo95_tj", "hi95_tj"]]
            .style.format({c: "{:,.0f}" for c in
                           ["forecast_tj", "lo80_tj", "hi80_tj", "lo95_tj", "hi95_tj"]}),
            use_container_width=True,
        )
    else:
        st.dataframe(fc[["year", "forecast_tj"]], use_container_width=True)


# ============================================================
# PAGE — COUNTRY DIAGNOSTICS
# ============================================================
elif page == "🩺 Country Diagnostics":
    st.header("Country Diagnostics")
    st.caption(
        "Pick a country to see its main household-energy challenges, "
        "with a side-by-side comparison against the European baseline."
    )

    country_total = load_csv(PROCESSED / "country_total.csv") if PROCESSED else None
    panel = load_csv(PROCESSED / "panel_level.csv") if PROCESSED else None
    forecasts = load_csv(PRED_DIR / "forecasts_2025_2027_with_ci.csv")
    if forecasts is None:
        forecasts = load_csv(PRED_DIR / "forecasts_2025_2027.csv")

    if country_total is None or panel is None:
        info_box(
            "<b>Diagnostic data not found.</b> Run <b>Phase 2</b> notebook first.",
            kind="warn",
        )
        st.stop()

    all_countries = sorted(country_total.country.unique())
    country = st.selectbox("Select a country", all_countries, index=0)

    c_hist = country_total[country_total.country == country].sort_values("year")
    if c_hist.empty:
        st.warning(f"No data for {country}.")
        st.stop()

    c_latest = c_hist.iloc[-1]
    c_avg = c_hist.energy_tj.mean()
    c_std = c_hist.energy_tj.std()
    c_cv = (c_std / c_avg) if c_avg > 0 else 0

    latest_year = int(c_latest.year)
    eu_latest = country_total[country_total.year == latest_year].groupby("country").energy_tj.mean()
    eu_avg = eu_latest.mean()

    growth = (
        (c_hist.energy_tj.iloc[-1] - c_hist.energy_tj.iloc[0]) / c_hist.energy_tj.iloc[0] * 100
        if len(c_hist) >= 2 and c_hist.energy_tj.iloc[0] > 0 else 0.0
    )

    # ---- KPI strip ----
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi(f"Latest ({latest_year})", f"{c_latest.energy_tj:,.0f} TJ", "bolt")
    with c2: kpi("Average 2010–2024", f"{c_avg:,.0f} TJ", "chart")
    with c3: kpi("Volatility (CV)", f"{c_cv:.2f}", "target")
    with c4: kpi("Growth since start", f"{growth:+.1f}%", "trend")

    st.markdown("---")

    # ============================================================
    # FUEL MIX
    # ============================================================
    df_country = panel[panel.country == country].copy()
    df_country_fuels = df_country[df_country.fuel_code != "TOTAL"].copy()

    if "end_use_code" in df_country_fuels.columns:
        mask_total = df_country_fuels.end_use_code == "FC_OTH_HH_E"
        if mask_total.any():
            df_country_fuels = df_country_fuels[mask_total]

    fuel_mix = (
        df_country_fuels.groupby("fuel_code")["energy_tj"].sum()
        .sort_values(ascending=False)
    )
    fuel_total = fuel_mix.sum() if len(fuel_mix) > 0 else 0
    fuel_pct = (fuel_mix / fuel_total * 100).round(2) if fuel_total > 0 else pd.Series(dtype=float)

    df_eu_fuels = panel[panel.fuel_code != "TOTAL"].copy()
    if "end_use_code" in df_eu_fuels.columns:
        mask_total_eu = df_eu_fuels.end_use_code == "FC_OTH_HH_E"
        if mask_total_eu.any():
            df_eu_fuels = df_eu_fuels[mask_total_eu]

    eu_fuel_mix = (
        df_eu_fuels.groupby("fuel_code")["energy_tj"].sum()
        .sort_values(ascending=False)
    )
    eu_fuel_total = eu_fuel_mix.sum() if len(eu_fuel_mix) > 0 else 0
    eu_fuel_pct = (eu_fuel_mix / eu_fuel_total * 100).round(2) if eu_fuel_total > 0 else pd.Series(dtype=float)

    # ============================================================
    # DETECT PROBLEMS
    # ============================================================
    st.subheader("🔍 Detected challenges")

    problems = []

    if c_cv > 0.15:
        problems.append({"severity": "High", "issue": "High consumption volatility", "evidence": f"CV = {c_cv:.2f} (>0.15)."})
    elif c_cv > 0.08:
        problems.append({"severity": "Medium", "issue": "Moderate volatility", "evidence": f"CV = {c_cv:.2f}."})

    if growth < -20:
        problems.append({"severity": "High", "issue": "Strong consumption decline", "evidence": f"Growth = {growth:+.1f}%."})
    elif growth < -5:
        problems.append({"severity": "Medium", "issue": "Gradual decline", "evidence": f"Growth = {growth:+.1f}%."})
    elif growth > 20:
        problems.append({"severity": "High", "issue": "Rapid consumption growth", "evidence": f"Growth = {growth:+.1f}%."})

    if len(fuel_pct) >= 1:
        top_fuel = fuel_pct.index[0]
        top_pct = float(fuel_pct.iloc[0])
        if top_pct > 70:
            problems.append({"severity": "High", "issue": f"Dominated by {top_fuel}", "evidence": f"{top_fuel} = {top_pct:.1f}%."})
        elif top_pct > 50:
            problems.append({"severity": "Medium", "issue": f"Concentrated ({top_fuel})", "evidence": f"{top_fuel} = {top_pct:.1f}%."})

    fossil_codes = {"G3000", "O4000", "O4630", "O4669", "O4671", "SFF_P1000_S2000"}
    renewable_codes = {"RA000", "RA410", "RA600", "R5110-5150_W6000RI", "R5300"}

    fossil_share = sum(fuel_pct.get(c, 0) for c in fossil_codes) if len(fuel_pct) > 0 else 0
    renew_share = sum(fuel_pct.get(c, 0) for c in renewable_codes) if len(fuel_pct) > 0 else 0

    if fossil_share > 60:
        problems.append({"severity": "High", "issue": "Fossil-fuel dependence", "evidence": f"Fossil share = {fossil_share:.1f}%."})
    elif fossil_share > 40:
        problems.append({"severity": "Medium", "issue": "Moderate fossil dependence", "evidence": f"Fossil share = {fossil_share:.1f}%."})

    if renew_share < 10 and fuel_total > 0:
        problems.append({"severity": "Medium", "issue": "Low renewable share", "evidence": f"Renewable share = {renew_share:.1f}%."})

    if eu_avg > 0:
        if c_latest.energy_tj > eu_avg * 2:
            problems.append({"severity": "Medium", "issue": "Above EU average", "evidence": f"×{c_latest.energy_tj/eu_avg:.1f} vs Europe."})
        elif c_latest.energy_tj < eu_avg * 0.5:
            problems.append({"severity": "Info", "issue": "Below EU average", "evidence": f"×{c_latest.energy_tj/eu_avg:.1f} vs Europe."})

    if not problems:
        info_box(f"<b>No major red flags detected for {country}.</b>", kind="success")
    else:
        for p in problems:
            kind_map = {"High": "danger", "Medium": "warn", "Info": "info"}
            info_box(f"<b>[{p['severity']}] {p['issue']}</b><br>{p['evidence']}", kind=kind_map.get(p["severity"], "info"))

    st.markdown("---")

    # ============================================================
    # FUEL MIX COMPARISON
    # ============================================================
    st.subheader("📊 Fuel mix — country vs Europe")

    if len(fuel_pct) == 0 and len(eu_fuel_pct) == 0:
        info_box("<b>No fuel-mix data available.</b>", kind="warn")
    else:
        all_codes = sorted(set(fuel_pct.index) | set(eu_fuel_pct.index))
        compare_rows = []
        for code in all_codes:
            compare_rows.append({
                "fuel_code": code,
                "country_pct": float(fuel_pct.get(code, 0.0)),
                "eu_pct": float(eu_fuel_pct.get(code, 0.0)),
            })

        compare_df = pd.DataFrame(compare_rows)
        if compare_df.empty:
            info_box("<b>No comparison data available.</b>", kind="warn")
        else:
            compare_df["gap"] = compare_df["country_pct"] - compare_df["eu_pct"]
            compare_df = compare_df.sort_values("country_pct", ascending=False).head(10)

            st.dataframe(
                compare_df.style.format({"country_pct": "{:.2f}%", "eu_pct": "{:.2f}%", "gap": "{:+.2f} pp"}),
                use_container_width=True, hide_index=True
            )

            fig = go.Figure()
            fig.add_trace(go.Bar(name=country, x=compare_df.fuel_code, y=compare_df.country_pct, marker_color=T["primary"]))
            fig.add_trace(go.Bar(name="Europe average", x=compare_df.fuel_code, y=compare_df.eu_pct, marker_color=T["text_mute"]))
            fig.update_layout(barmode="group", height=420, xaxis_title="Fuel code", yaxis_title="% of fuel mix")
            st.plotly_chart(styled_plotly(fig), use_container_width=True)

    # ============================================================
    # END-USE COMPARISON
    # ============================================================
    if "end_use_short" in panel.columns:
        st.subheader("📊 End-use mix — country vs Europe")

        enduse_source = load_csv(PROCESSED / "households_by_enduse.csv")
        
        if enduse_source is not None and "end_use_short" in enduse_source.columns:
            df_country_eu = enduse_source[(enduse_source.country == country) & (enduse_source.end_use_short != "total")]
            df_eu_enduse = enduse_source[enduse_source.end_use_short != "total"]
        else:
            df_country_eu = panel[(panel.country == country) & (panel.end_use_short != "total")]
            df_eu_enduse = panel[panel.end_use_short != "total"]

        if df_country_eu.empty or df_eu_enduse.empty:
            info_box("<b>End-use data not available.</b>", kind="warn")
        else:
            c_mix = df_country_eu.groupby("end_use_short")["energy_tj"].sum()
            c_total = c_mix.sum()
            c_pct = (c_mix / c_total * 100).round(2) if c_total > 0 else pd.Series(dtype=float)

            e_mix = df_eu_enduse.groupby("end_use_short")["energy_tj"].sum()
            e_total = e_mix.sum()
            e_pct = (e_mix / e_total * 100).round(2) if e_total > 0 else pd.Series(dtype=float)

            all_enduse = sorted(set(c_pct.index) | set(e_pct.index))
            rows = [{"end_use": eu, "country_pct": float(c_pct.get(eu, 0.0)), "eu_pct": float(e_pct.get(eu, 0.0))} for eu in all_enduse]
            
            enduse_df = pd.DataFrame(rows)
            enduse_df["gap"] = enduse_df["country_pct"] - enduse_df["eu_pct"]

            st.dataframe(
                enduse_df.style.format({"country_pct": "{:.2f}%", "eu_pct": "{:.2f}%", "gap": "{:+.2f} pp"}),
                use_container_width=True, hide_index=True
            )

            fig = go.Figure()
            fig.add_trace(go.Bar(name=country, x=enduse_df.end_use, y=enduse_df.country_pct, marker_color=T["primary"]))
            fig.add_trace(go.Bar(name="Europe average", x=enduse_df.end_use, y=enduse_df.eu_pct, marker_color=T["text_mute"]))
            fig.update_layout(barmode="group", height=420, xaxis_title="End-use", yaxis_title="% of mix")
            st.plotly_chart(styled_plotly(fig), use_container_width=True)

    # ============================================================
    # HISTORICAL TREND + FORECAST
    # ============================================================
    if forecasts is not None:
        st.subheader("📈 Historical trend + forecast")
        fc = forecasts[forecasts.country == country].sort_values("year")
        if not fc.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=c_hist.year, y=c_hist.energy_tj, mode="lines+markers", name="Historical", line=dict(color=T["primary"], width=3)))
            fig.add_trace(go.Scatter(x=fc.year, y=fc.forecast_tj, mode="lines+markers", name="Forecast", line=dict(color=T["accent"], dash="dash", width=3)))
            if "lo80_tj" in fc.columns:
                fig.add_trace(go.Scatter(x=pd.concat([fc.year, fc.year[::-1]]), y=pd.concat([fc.hi95_tj, fc.lo95_tj[::-1]]), fill="toself", fillcolor="rgba(245,158,11,0.15)", line=dict(color="rgba(255,255,255,0)"), name="95% CI"))
            fig.update_layout(height=400, xaxis_title="Year", yaxis_title="Energy (TJ)", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(styled_plotly(fig), use_container_width=True)


# ============================================================
# PAGE — CLUSTERING
# ============================================================
elif page == "🎯 Clustering":
    st.header("Country Clusters")
    st.caption("KMeans segmentation of European countries by household energy profile.")

    bundle = load_clustering_bundle()
    clusters = load_csv(REPORTS / "phase5_clusters.csv")
    profile = load_csv(REPORTS / "phase5_cluster_profile.csv")

    if bundle is None or clusters is None:
        info_box(
            "<b>Clustering artifacts not found.</b> Run <b>Phase 5</b> notebook first.",
            kind="warn",
        )
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Countries", len(clusters), "globe")
    with c2: kpi("Clusters (k)", bundle.get("k", "—"), "target")
    with c3: kpi("Features", len(bundle.get("features", [])), "grid")
    with c4: kpi("Algorithm", "KMeans", "cog")

    st.markdown("---")
    st.subheader("PCA 2D projection")
    show_figure("outputs/figures/phase5_pca_clusters.png")

    st.subheader("Cluster profile")
    if profile is not None:
        st.dataframe(profile, use_container_width=True)

    st.subheader("Country lookup")
    country = st.selectbox("Select a country", sorted(clusters.country.unique()), index=0)
    row = clusters[clusters.country == country].iloc[0]
    info_box(
        f"<b>{country}</b> belongs to <b>Cluster {row['cluster_kmeans']}</b> — "
        f"<i>{row.get('cluster_label', 'Unlabeled')}</i>"
    )


# ============================================================
# PAGE — MODEL REPORT
# ============================================================
elif page == "📋 Model Report":
    st.header("Model Report")
    st.caption("Consolidated evaluation artifacts.")

    st.subheader("Prediction (Phase 3)")
    comp = load_csv(REPORTS / "phase3_model_comparison.csv")
    if comp is not None: st.dataframe(comp, use_container_width=True)
    imp = load_csv(REPORTS / "phase3_feature_importance.csv")
    if imp is not None: st.dataframe(imp, use_container_width=True)
    show_figure("outputs/figures/phase3_feature_importance.png")

    st.subheader("Forecasting (Phase 4 + Phase 6)")
    bt = load_csv(REPORTS / "phase4_backtest_results.csv")
    if bt is not None: st.dataframe(bt, use_container_width=True)
    bp = load_csv(REPORTS / "phase4_best_model_per_country.csv")
    if bp is not None: st.dataframe(bp, use_container_width=True)
    mc = load_csv(REPORTS / "phase6_model_comparison.csv")
    if mc is not None: st.dataframe(mc, use_container_width=True)
    mh = load_csv(REPORTS / "phase6_model_by_horizon.csv")
    if mh is not None: st.dataframe(mh, use_container_width=True)

    st.subheader("Clustering (Phase 5)")
    cprof = load_csv(REPORTS / "phase5_cluster_profile.csv")
    if cprof is not None: st.dataframe(cprof, use_container_width=True)

st.markdown("---")
st.caption("Eurostat data (CC-BY) · Built with Python & Streamlit")