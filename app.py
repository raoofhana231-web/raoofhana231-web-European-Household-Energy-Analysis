# ============================================================
# European Household Energy — Professional ML Dashboard
# Auto file discovery · Light/Dark · Fallback forecasting
# Run: streamlit run app.py
# ============================================================
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ------------------------------------------------------------
# 1. APP CONFIG (must be first Streamlit call)
# ------------------------------------------------------------
st.set_page_config(
    page_title="European Household Energy — ML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# 2. PATH RESOLUTION — recursive auto-discovery
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {"node_modules", "venv", ".venv", "env", "__pycache__",
             "site-packages", ".ipynb_checkpoints", "$RECYCLE.BIN"}
EXTS = (".csv", ".joblib", ".png", ".pkl")

with st.sidebar:
    st.markdown(
        '<div style="padding:8px 0 16px 0;">'
        '<div style="font-size:20px;font-weight:800;letter-spacing:-0.02em;">⚡ Energy ML</div>'
        '<div style="font-size:11px;opacity:.7;letter-spacing:.08em;'
        'text-transform:uppercase;">European Households</div></div>',
        unsafe_allow_html=True,
    )
    theme_choice = st.radio("Appearance", ["☀️ Light", "🌙 Dark", "🖥️ Auto"], index=2)
    custom_root_str = st.text_input(
        "Custom project folder (optional)", value="",
        help="Paste the full path of your project folder if files are not found automatically.",
    ).strip().strip('"')


@st.cache_data(show_spinner=False)
def _scan(root_str: str, max_depth: int):
    """Index every csv/joblib/png/pkl under root_str -> {filename: [paths]}."""
    found = {}
    if not root_str or not os.path.isdir(root_str):
        return found
    base_depth = root_str.rstrip(os.sep).count(os.sep)
    for dp, dn, fn in os.walk(root_str):
        dn[:] = [d for d in dn if d not in SKIP_DIRS and not d.startswith(".")]
        if dp.count(os.sep) - base_depth >= max_depth:
            dn[:] = []
        for f in fn:
            if f.lower().endswith(EXTS):
                found.setdefault(f, []).append(os.path.join(dp, f))
    return found


def _search_roots():
    roots = []
    if custom_root_str:
        roots.append((custom_root_str, 10))
    roots.append((str(ROOT), 10))
    roots.append((str(ROOT.parent), 4))
    return roots


def locate(filename: str):
    """Return newest Path matching filename anywhere in the search roots, else None."""
    candidates = []
    for r, depth in _search_roots():
        candidates += _scan(r, depth).get(filename, [])
    if not candidates:
        return None
    candidates = list(dict.fromkeys(candidates))
    return Path(max(candidates, key=lambda p: os.path.getmtime(p)))


def P(filename: str) -> Path:
    """Path of a file (found anywhere) or a non-existing placeholder path."""
    return locate(filename) or (ROOT / "__missing__" / filename)


PANEL_PATH = locate("panel_level.csv")
CT_PATH = locate("country_total.csv")
PROCESSED = (PANEL_PATH or CT_PATH).parent if (PANEL_PATH or CT_PATH) else None

# ------------------------------------------------------------
# 3. THEME TOKENS
# ------------------------------------------------------------
LIGHT = {
    "bg": "#f8fafc", "bg_soft": "#f1f5f9", "surface": "#ffffff", "border": "#e2e8f0",
    "text": "#0f172a", "text_mute": "#64748b", "primary": "#4f46e5", "primary_2": "#6366f1",
    "accent": "#10b981", "danger": "#ef4444", "warning": "#f59e0b",
    "shadow": "rgba(15,23,42,0.06)", "shadow_2": "rgba(15,23,42,0.10)",
    "chart_bg": "#ffffff", "chart_fg": "#0f172a", "chart_grid": "#e2e8f0",
}
DARK = {
    "bg": "#0b1120", "bg_soft": "#111827", "surface": "#1e293b", "border": "#334155",
    "text": "#f1f5f9", "text_mute": "#94a3b8", "primary": "#818cf8", "primary_2": "#a5b4fc",
    "accent": "#34d399", "danger": "#f87171", "warning": "#fbbf24",
    "shadow": "rgba(0,0,0,0.35)", "shadow_2": "rgba(0,0,0,0.55)",
    "chart_bg": "#1e293b", "chart_fg": "#f1f5f9", "chart_grid": "#334155",
}
is_dark = theme_choice == "🌙 Dark"
T = DARK if is_dark else LIGHT

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
        background: {T['surface']}; border: 1px solid {T['border']}; border-radius: 14px;
        padding: 20px 22px; box-shadow: 0 1px 2px {T['shadow']}, 0 4px 16px {T['shadow']};
        position: relative; transition: all .2s cubic-bezier(.4,0,.2,1); height: 100%;
    }}
    .kpi:hover {{
        border-color: {T['primary']}40;
        box-shadow: 0 4px 12px {T['shadow_2']}, 0 8px 32px {T['shadow']};
        transform: translateY(-2px);
    }}
    .kpi-icon {{
        width: 36px; height: 36px; border-radius: 10px; background: {T['primary']}14;
        color: {T['primary']}; display: flex; align-items: center; justify-content: center;
        margin-bottom: 14px; font-size: 18px; font-weight: 700;
    }}
    .kpi-label {{ font-size: 11px; color: {T['text_mute']}; text-transform: uppercase;
        letter-spacing: 0.08em; font-weight: 600; margin-bottom: 6px; }}
    .kpi-value {{ font-size: 26px; font-weight: 800; color: {T['text']}; line-height: 1.15;
        letter-spacing: -0.02em; }}
    .kpi-delta {{ font-size: 12px; margin-top: 8px; font-weight: 600; color: {T['accent']}; }}
    .insight {{ background: {T['primary']}08; border: 1px solid {T['primary']}25;
        border-left: 4px solid {T['primary']}; padding: 16px 20px; border-radius: 10px;
        margin: 16px 0; font-size: 14px; line-height: 1.7; color: {T['text']}; }}
    .warn {{ background: {T['warning']}10; border: 1px solid {T['warning']}30;
        border-left: 4px solid {T['warning']}; padding: 16px 20px; border-radius: 10px;
        margin: 16px 0; color: {T['text']}; }}
    .success {{ background: {T['accent']}10; border: 1px solid {T['accent']}30;
        border-left: 4px solid {T['accent']}; padding: 16px 20px; border-radius: 10px;
        margin: 16px 0; color: {T['text']}; }}
    .danger {{ background: {T['danger']}10; border: 1px solid {T['danger']}30;
        border-left: 4px solid {T['danger']}; padding: 16px 20px; border-radius: 10px;
        margin: 16px 0; color: {T['text']}; }}
    section[data-testid="stSidebar"] {{ background: {T['surface']} !important;
        border-right: 1px solid {T['border']}; }}
    section[data-testid="stSidebar"] * {{ color: {T['text']} !important; }}
    section[data-testid="stSidebar"] .stRadio > div {{ gap: 4px; }}
    section[data-testid="stSidebar"] .stRadio label {{
        padding: 9px 14px !important; border-radius: 10px !important; cursor: pointer;
        transition: all .15s ease; font-size: 14px; font-weight: 500;
        color: {T['text_mute']} !important; }}
    section[data-testid="stSidebar"] .stRadio label:hover {{
        background: {T['primary']}0d; color: {T['text']} !important; }}
    .stButton > button, .stFormSubmitButton > button {{
        background: {T['primary']}; color: white !important; border: none; border-radius: 10px;
        padding: 10px 20px; font-weight: 600; font-size: 14px; transition: all .15s ease;
        box-shadow: 0 1px 2px {T['shadow']}; }}
    .stButton > button:hover, .stFormSubmitButton > button:hover {{
        background: {T['primary_2']}; box-shadow: 0 4px 12px {T['primary']}40;
        transform: translateY(-1px); }}
    .stTextInput > div > div > input, .stSelectbox > div > div,
    .stNumberInput > div > div > input {{
        background: {T['surface']} !important; color: {T['text']} !important;
        border: 1px solid {T['border']} !important; border-radius: 10px !important; }}
    .stSlider > div > div > div {{ background: {T['primary']} !important; }}
    .stDataFrame {{ border-radius: 12px; overflow: hidden; border: 1px solid {T['border']};
        background: {T['surface']}; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {T['border']};
        background: transparent; }}
    .stTabs [data-baseweb="tab"] {{ background: transparent; border-radius: 8px 8px 0 0;
        padding: 10px 18px; color: {T['text_mute']}; font-weight: 500; }}
    .stTabs [aria-selected="true"] {{ color: {T['primary']} !important;
        background: {T['primary']}08 !important; border-bottom: 2px solid {T['primary']} !important; }}
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
        "globe": '<path d="M12 2a10 10 0 100 20 10 10 0 000-20zm0 2a8 8 0 017.93 7H15.9a12 12 0 00-2.9-5.66A8 8 0 0112 4zm-2.86.34A12 12 0 006.24 9H4.07a8 8 0 014.99-4.66zM4.07 15h2.17a12 12 0 002.9 5.66A8 8 0 014.07 15zm4.17 0h7.52a10 10 0 01-3.76 5.4A10 10 0 018.24 15zm7.52-2H8.24a10 10 0 013.76-5.4A10 10 0 0115.76 13zm1.17 2h2.17a8 8 0 01-4.99 4.66A12 12 0 0016.93 15zm2.17-2h-2.17a12 12 0 00-2.9 5.66A8 8 0 0019.1 13z"/>',
        "calendar": '<path d="M19 3h-1V1h-2v2H8V1H6v2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V8h14v11z"/>',
        "database": '<path d="M12 3C7.58 3 4 4.79 4 7v10c0 2.21 3.58 4 8 4s8-1.79 8-4V7c0-2.21-3.58-4-8-4zm0 2c3.87 0 6 1.5 6 2s-2.13 2-6 2-6-1.5-6-2 2.13-2 6-2zm6 12c0 .5-2.13 2-6 2s-6-1.5-6-2v-2.23c1.61.78 3.72 1.23 6 1.23s4.39-.45 6-1.23V17zm0-5c0 .5-2.13 2-6 2s-6-1.5-6-2V9.77C7.61 10.55 9.72 11 12 11s4.39-.45 6-1.23V12z"/>',
        "chart": '<path d="M3 13h2v8H3v-8zm4-4h2v12H7V9zm4-6h2v18h-2V3zm4 8h2v10h-2V11zm4-4h2v14h-2V7z"/>',
        "bolt": '<path d="M11 21h-1l1-7H7.5c-.88 0-.33-.75-.31-.78C8.48 10.94 10.42 7.54 13.01 3h1l-1 7h3.51c.4 0 .62.19.4.66C12.97 17.55 11 21 11 21z"/>',
        "target": '<path d="M12 2a10 10 0 100 20 10 10 0 000-20zm0 18a8 8 0 110-16 8 8 0 010 16zm0-13a5 5 0 100 10 5 5 0 000-10zm0 8a3 3 0 110-6 3 3 0 010 6z"/>',
        "cog": '<path d="M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>',
        "sparkles": '<path d="M19 9l1.25-2.75L23 5l-2.75-1.25L19 1l-1.25 2.75L15 5l2.75 1.25L19 9zm-7.5.5L9 4 6.5 9.5 1 12l5.5 2.5L9 20l2.5-5.5L17 12l-5.5-2.5zM19 15l-1.25 2.75L15 19l2.75 1.25L19 23l1.25-2.75L23 19l-2.75-1.25L19 15z"/>',
        "trend": '<path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/>',
        "grid": '<path d="M3 3h8v8H3V3zm10 0h8v8h-8V3zM3 13h8v8H3v-8zm10 0h8v8h-8v-8z"/>',
        "stethoscope": '<path d="M19 8h-2V3h-2v5h-2V3h-2v6c0 1.66 1.34 3 3 3h1v3.5a3.5 3.5 0 01-7 0V9c0-1.1-.9-2-2-2s-2 .9-2 2v6.5C4 19.54 6.46 22 9.5 22s5.5-2.46 5.5-5.5V12h4v-2z"/>',
    }
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'width="{size}" height="{size}" fill="{c}">{icons.get(name, icons["chart"])}</svg>'
    )


# ------------------------------------------------------------
# 5. LOADERS
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_prediction_bundle(adv_path: str, model_path: str, prep_path: str):
    if os.path.exists(adv_path):
        try:
            bundle = joblib.load(adv_path)
            return {"type": "advanced", "bundle": bundle,
                    "name": bundle.get("model_name", "Advanced")}
        except Exception:
            pass
    if os.path.exists(model_path) and os.path.exists(prep_path):
        try:
            model_obj = joblib.load(model_path)
            preprocessor = joblib.load(prep_path)
            return {"type": "legacy", "model": model_obj["model"],
                    "name": model_obj.get("name", "HistGB"), "preprocessor": preprocessor}
        except Exception:
            pass
    return None


@st.cache_resource(show_spinner=False)
def load_clustering_bundle(path: str):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_csv(path):
    if path and Path(path).exists():
        return pd.read_csv(path)
    return None


def get_panel():
    return load_csv(PANEL_PATH) if PANEL_PATH else None


def get_country_total():
    ct = load_csv(CT_PATH) if CT_PATH else None
    if ct is not None:
        return ct
    panel = get_panel()
    if panel is None or "energy_tj" not in panel.columns:
        return None
    p = panel
    if "fuel_code" in p.columns and (p.fuel_code == "TOTAL").any():
        p = p[p.fuel_code == "TOTAL"]
    if "end_use_short" in p.columns and (p.end_use_short == "total").any():
        p = p[p.end_use_short == "total"]
    return p.groupby(["country", "year"], as_index=False)["energy_tj"].sum()


def build_trend_forecast(ct: pd.DataFrame) -> pd.DataFrame:
    """Fallback: per-country linear trend (last 8 yrs) with widening intervals."""
    rows = []
    for c, g in ct.groupby("country"):
        g = g.groupby("year", as_index=False)["energy_tj"].sum().sort_values("year").tail(8)
        if len(g) < 3:
            continue
        x = g.year.values.astype(float)
        y = g.energy_tj.values.astype(float)
        slope, inter = np.polyfit(x, y, 1)
        s = float((y - (slope * x + inter)).std(ddof=1))
        for yr in range(2025, 2030):
            h = max(yr - int(x.max()), 1)
            f = max(0.0, slope * yr + inter)
            w = s * np.sqrt(h)
            rows.append(dict(country=c, year=yr, forecast_tj=f,
                             lo80_tj=max(0.0, f - 1.28 * w), hi80_tj=f + 1.28 * w,
                             lo95_tj=max(0.0, f - 1.96 * w), hi95_tj=f + 1.96 * w))
    return pd.DataFrame(rows)


def load_forecast_source():
    # 1) Advanced 5-year forecasts
    p = locate("country_forecasts_2025_2029.csv")
    if p is not None:
        df = pd.read_csv(p)
        if {"country", "year", "forecast_tj"}.issubset(df.columns):
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df["forecast_tj"] = pd.to_numeric(df["forecast_tj"], errors="coerce")
            df = df.dropna(subset=["country", "year", "forecast_tj"])
            df["year"] = df["year"].astype(int)
            df = df[df["year"].between(2025, 2029)].sort_values(["country", "year"])
            if not df.empty:
                return df, "Advanced Forecasting (2025–2029)"
    # 2) Legacy forecasts
    for name in ("forecasts_2025_2027_with_ci.csv", "forecasts_2025_2027.csv"):
        p = locate(name)
        if p is not None:
            df = load_csv(p)
            if df is not None and not df.empty:
                return df, "Phase 4/6 legacy forecasts"
    # 3) Auto-generated fallback from historical data
    ct = get_country_total()
    if ct is not None and not ct.empty:
        df = build_trend_forecast(ct)
        if not df.empty:
            return df, "Auto-generated trend forecast (fallback)"
    return None, "none"


# ------------------------------------------------------------
# 6. UI HELPERS
# ------------------------------------------------------------
def kpi(label, value, icon_name="chart", delta=None):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="kpi"><div class="kpi-icon">{icon(icon_name, 18)}</div>'
        f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
        f'{delta_html}</div>',
        unsafe_allow_html=True,
    )


def info_box(text, kind="info"):
    cls = {"info": "insight", "warn": "warn", "success": "success", "danger": "danger"}[kind]
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def show_figure(relpath, caption=None):
    p = locate(Path(relpath).name)
    if p is not None:
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.caption(f"Figure not available: {Path(relpath).name}")


def styled_plotly(fig):
    fig.update_layout(
        paper_bgcolor=T["chart_bg"], plot_bgcolor=T["chart_bg"],
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
# 7. HEADER + SIDEBAR INFO
# ------------------------------------------------------------
st.markdown(
    f"""
    <div style="margin-bottom:24px;">
        <div style="display:flex;align-items:center;gap:14px;">
            <div style="width:44px;height:44px;border-radius:12px;background:{T['primary']}14;
                        color:{T['primary']};display:flex;align-items:center;justify-content:center;">
                {icon("bolt", 22)}
            </div>
            <div>
                <div style="font-size:26px;font-weight:800;letter-spacing:-0.02em;color:{T['text']};">
                    Household Energy Intelligence</div>
                <div style="font-size:13px;color:{T['text_mute']};margin-top:2px;">
                    European countries · 2010–2024 · ML dashboard</div>
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
            f'<div style="font-size:11px;color:{T["text_mute"]};text-transform:uppercase;'
            f'letter-spacing:.08em;margin-bottom:6px;">Data source</div>'
            f'<code style="font-size:11px;color:{T["text"]};background:{T["bg_soft"]};'
            f'padding:4px 8px;border-radius:6px;display:block;word-break:break-all;">{PROCESSED}</code>',
            unsafe_allow_html=True,
        )
    else:
        st.error("Processed data not found")
    with st.expander("Debug: files found"):
        st.write(f"App folder: `{ROOT}`")
        idx = {}
        for r, d in _search_roots():
            for k, v in _scan(r, d).items():
                idx.setdefault(k, []).extend(v)
        csvs = sorted(k for k in idx if k.endswith(".csv"))
        st.write(f"CSV files found: {len(csvs)}")
        st.write(csvs[:60] if csvs else "None — set the custom folder above.")

PAGES = ["🏠 Home", "📊 Data Overview", "🔮 Prediction", "📈 Forecasting",
         "🩺 Country Diagnostics", "🎯 Clustering", "📋 Model Report"]
page = st.sidebar.radio("Navigation", PAGES, index=0, label_visibility="collapsed")


def stop_no_panel():
    info_box(
        "<b>Panel data not found.</b> <code>panel_level.csv</code> was not found anywhere under "
        f"<code>{ROOT}</code> (or its parent). Run the <b>Phase 2</b> notebook to generate it, "
        "or paste your project folder path in the sidebar (Custom project folder).",
        kind="warn",
    )
    st.stop()


# ============================================================
# PAGE — HOME
# ============================================================
if page == "🏠 Home":
    country_total = get_country_total()
    panel = get_panel()
    forecasts, forecast_source = load_forecast_source()

    n_countries = int(country_total.country.nunique()) if country_total is not None else 0
    n_years = int(country_total.year.nunique()) if country_total is not None else 0
    n_panel = len(panel) if panel is not None else 0
    n_forecast = len(forecasts) if forecasts is not None else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Countries", n_countries, "globe")
    with c2: kpi("Years covered", n_years, "calendar")
    with c3: kpi("Panel observations", f"{n_panel:,}", "database")
    with c4: kpi("Forecasts generated", n_forecast, "sparkles")

    st.markdown("---")
    st.subheader("What this dashboard shows")

    def card(ic, title, desc):
        st.markdown(
            f'<div class="kpi" style="height:auto;"><div class="kpi-icon">{icon(ic, 18)}</div>'
            f'<div style="font-weight:700;font-size:15px;margin-bottom:6px;">{title}</div>'
            f'<div style="font-size:13px;color:{T["text_mute"]};">{desc}</div></div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns(2)
    with left:
        card("bolt", "Prediction", "Point prediction of household energy for a given country × year × end-use × fuel.")
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        card("trend", "Forecasting", "Per-country 2025–2029 forecasts with confidence intervals.")
    with right:
        card("stethoscope", "Country Diagnostics", "Pick a country → see its energy challenges + comparison with Europe.")
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        card("target", "Clustering", "KMeans segmentation of countries by household energy profile.")

    if panel is None:
        info_box("<b>Data not found.</b> Run <b>Phase 2</b> notebook to generate "
                 "<code>panel_level.csv</code>, or set the custom folder in the sidebar.", kind="warn")
    else:
        info_box("<b>Dataset:</b> Eurostat household energy, 33,874 raw rows. Cleaned to "
                 "15,813 panel rows × 35 end-use/fuel combinations × 30 countries.")
        if forecasts is not None:
            st.caption(f"Forecast source: {forecast_source}")


# ============================================================
# PAGE — DATA OVERVIEW
# ============================================================
elif page == "📊 Data Overview":
    st.header("Data Overview")
    panel = get_panel()
    country_total = get_country_total()
    if panel is None:
        stop_no_panel()

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
                             color_continuous_scale=[[0, T["bg_soft"]], [1, T["primary"]]], height=700)
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
# PAGE — PREDICTION
# ============================================================
elif page == "🔮 Prediction":
    st.header("Energy Consumption Prediction")
    st.caption("Predict household energy for a country × year × end-use × fuel.")

    bundle = load_prediction_bundle(
        str(P("advanced_prediction_bundle.joblib")),
        str(P("best_prediction_model.joblib")),
        str(P("preprocessor.joblib")),
    )
    panel = get_panel()
    comparison = load_csv(locate("phase3_model_comparison.csv"))

    if panel is None:
        stop_no_panel()
    if bundle is None:
        info_box("<b>Model not found.</b> Expected <code>advanced_prediction_bundle.joblib</code> "
                 "or <code>best_prediction_model.joblib</code> (+ <code>preprocessor.joblib</code>). "
                 "Run the prediction notebook first.", kind="warn")
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
            hist = panel[(panel.country == country) & (panel.end_use_short == end_use) &
                         (panel.fuel_code == fuel)].sort_values("year")

            if hist.empty:
                st.warning(f"⚠️ No exact historical data for **{country}** × **{end_use}** × **{fuel}**. Using fallback baseline...")
                fa = panel[(panel.country == country) & (panel.fuel_code == fuel)].sort_values("year")
                fb = panel[(panel.country == country) & (panel.end_use_short == end_use)].sort_values("year")
                fc_ = panel[panel.country == country].sort_values("year")
                if not fa.empty: hist = fa
                elif not fb.empty: hist = fb
                elif not fc_.empty: hist = fc_
                else:
                    st.error(f"Completely missing data for {country}.")
                    st.stop()

            hist_before = hist[hist.year < year]

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
                lag_2 = lag_3 = lag_1
            else:
                lag_1 = float(hist.iloc[0]["energy_tj"])
                lag_2 = lag_3 = lag_1

            rolling_mean = (lag_1 + lag_2 + lag_3) / 3
            rolling_std = float(np.std([lag_1, lag_2, lag_3])) if len(hist_before) >= 2 else 0.0

            if bundle["type"] == "advanced":
                b = bundle["bundle"]
                end_use_code_val = panel[panel.end_use_short == end_use]["end_use_code"].iloc[0]
                year_index = int(year) - b["min_year"]
                X_input = pd.DataFrame([{
                    "country": country, "end_use_code": end_use_code_val, "fuel_code": fuel,
                    "year": int(year), "year_index": year_index, "year_sq": year_index ** 2,
                    "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3,
                    "rolling_mean_3": rolling_mean, "rolling_std_3": rolling_std,
                }])
                y_tj = max(0.0, float(b["model"].predict(X_input[b["features"]])[0]))
            else:
                X_input = pd.DataFrame([{
                    "country": country, "end_use_short": end_use, "fuel_code": fuel, "year": year,
                    "prev_year_energy": lag_1, "prev_2yr_energy": lag_2, "rolling_2_prev": rolling_mean,
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

            info_box(f"<b>{country}</b> · <b>{year}</b> · <b>{end_use}</b> · <b>{fuel}</b><br>"
                     f"Predicted household energy: "
                     f"<b style='color:{T['primary']};font-size:18px;'>{y_tj:,.0f} TJ</b>")

            st.markdown("#### Historical context + prediction")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.year, y=hist.energy_tj, mode="lines+markers",
                                     name="Historical", line=dict(color=T["primary"], width=2.5),
                                     marker=dict(size=8)))
            fig.add_trace(go.Scatter(x=[year], y=[y_tj], mode="markers", name="Prediction",
                                     marker=dict(size=16, color=T["accent"], symbol="star",
                                                 line=dict(color=T["text"], width=1))))
            fig.add_vline(x=float(hist.year.max()) + 0.5, line_dash="dot", line_color=T["text_mute"])
            fig.update_layout(height=400, xaxis_title="Year", yaxis_title="Energy (TJ)",
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(styled_plotly(fig), use_container_width=True)

            last_val = float(hist_before.iloc[-1]["energy_tj"]) if len(hist_before) > 0 else y_tj
            delta_pct = (y_tj - last_val) / last_val * 100 if last_val > 0 else 0
            info_box(f"Compared to last observed (<b>{last_val:,.0f} TJ</b>): "
                     f"<b style='color:{T['accent'] if delta_pct >= 0 else T['danger']};'>"
                     f"{delta_pct:+.1f}%</b>")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            with st.expander("Debug info"):
                st.write(f"Model Type: {bundle['type']}")
                st.write(f"Model Name: {bundle['name']}")

    st.markdown("---")
    st.subheader("Model comparison (Phase 3)")
    if comparison is not None:
        st.dataframe(comparison, use_container_width=True)

    st.subheader("Feature importance")
    show_figure("phase3_feature_importance.png")


# ============================================================
# PAGE — FORECASTING
# ============================================================
elif page == "📈 Forecasting":
    st.header("Household Energy Forecasting (2025–2029)")
    st.caption("Country forecasts for 2025–2029. Confidence intervals are displayed when available.")

    country_total = get_country_total()
    forecasts, forecast_source = load_forecast_source()

    model_comp = load_csv(locate("phase6_model_comparison.csv"))
    model_h = load_csv(locate("phase6_model_by_horizon.csv"))
    best_per = load_csv(locate("phase4_best_model_per_country.csv"))

    if country_total is None or forecasts is None:
        info_box("<b>Forecasting data not found.</b> Neither forecast files nor historical "
                 "<code>country_total.csv</code>/<code>panel_level.csv</code> were found. "
                 "Set the custom project folder in the sidebar.", kind="warn")
        st.stop()

    st.caption(f"Forecast source: {forecast_source}")
    if forecast_source.startswith("Auto-generated"):
        info_box("No forecast file was found, so a simple <b>linear-trend forecast</b> was generated "
                 "from historical data. Run the forecasting notebook to export model-based forecasts.",
                 kind="info")

    has_ci = all(c in forecasts.columns for c in ["lo80_tj", "hi80_tj", "lo95_tj", "hi95_tj"])

    summary = forecasts.groupby("year").agg(total=("forecast_tj", "sum")).reset_index()
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Countries", int(forecasts.country.nunique()), "globe")
    with c2: kpi("Horizon", f"{int(forecasts.year.min())}–{int(forecasts.year.max())}", "calendar")
    with c3: kpi(f"{int(summary.year.iloc[-1])} total", f"{summary.total.iloc[-1]:,.0f} TJ", "bolt")
    with c4: kpi("CI available", "Yes" if has_ci else "No", "target")

    if model_comp is not None:
        st.subheader("Model comparison (Phase 6)")
        st.dataframe(model_comp, use_container_width=True)
    if model_h is not None:
        st.subheader("WAPE by horizon step (Phase 6)")
        st.dataframe(model_h, use_container_width=True)

    st.subheader("Country forecast")
    country = st.selectbox("Select a country", sorted(forecasts.country.unique()), index=0)

    hist = (country_total[country_total.country == country]
            .groupby("year", as_index=False)["energy_tj"].sum().sort_values("year"))
    fc = forecasts[forecasts.country == country].sort_values("year")

    best_model_name = forecast_source
    if best_per is not None and "country" in best_per.columns and "best_model" in best_per.columns:
        row = best_per[best_per.country == country]
        if not row.empty:
            best_model_name = row.iloc[0]["best_model"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.year, y=hist.energy_tj, mode="lines+markers", name="Historical",
                             line=dict(color=T["primary"], width=3), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=fc.year, y=fc.forecast_tj, mode="lines+markers",
                             name=f"Forecast ({best_model_name})",
                             line=dict(color=T["accent"], dash="dash", width=3),
                             marker=dict(size=10, symbol="diamond")))
    if has_ci:
        fig.add_trace(go.Scatter(x=pd.concat([fc.year, fc.year[::-1]]),
                                 y=pd.concat([fc.hi95_tj, fc.lo95_tj[::-1]]),
                                 fill="toself", fillcolor="rgba(245,158,11,0.15)",
                                 line=dict(color="rgba(255,255,255,0)"), name="95% CI"))
        fig.add_trace(go.Scatter(x=pd.concat([fc.year, fc.year[::-1]]),
                                 y=pd.concat([fc.hi80_tj, fc.lo80_tj[::-1]]),
                                 fill="toself", fillcolor="rgba(245,158,11,0.30)",
                                 line=dict(color="rgba(255,255,255,0)"), name="80% CI"))
    if len(hist):
        fig.add_vline(x=float(hist.year.max()) + 0.5, line_dash="dot", line_color=T["text_mute"])
    fig.update_layout(height=500, xaxis_title="Year", yaxis_title="Energy (TJ)",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(styled_plotly(fig), use_container_width=True)

    if has_ci:
        st.dataframe(
            fc[["year", "forecast_tj", "lo80_tj", "hi80_tj", "lo95_tj", "hi95_tj"]]
            .style.format({c: "{:,.0f}" for c in ["forecast_tj", "lo80_tj", "hi80_tj", "lo95_tj", "hi95_tj"]}),
            use_container_width=True)
    else:
        st.dataframe(fc[["year", "forecast_tj"]], use_container_width=True)


# ============================================================
# PAGE — COUNTRY DIAGNOSTICS
# ============================================================
elif page == "🩺 Country Diagnostics":
    st.header("Country Diagnostics")
    st.caption("Pick a country to see its main household-energy challenges, "
               "with a side-by-side comparison against the European baseline.")

    country_total = get_country_total()
    panel = get_panel()
    forecasts, _fsrc = load_forecast_source()

    if country_total is None or panel is None:
        stop_no_panel()

    country_total = country_total.groupby(["country", "year"], as_index=False)["energy_tj"].sum()
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

    growth = ((c_hist.energy_tj.iloc[-1] - c_hist.energy_tj.iloc[0]) / c_hist.energy_tj.iloc[0] * 100
              if len(c_hist) >= 2 and c_hist.energy_tj.iloc[0] > 0 else 0.0)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi(f"Latest ({latest_year})", f"{c_latest.energy_tj:,.0f} TJ", "bolt")
    with c2: kpi("Average 2010–2024", f"{c_avg:,.0f} TJ", "chart")
    with c3: kpi("Volatility (CV)", f"{c_cv:.2f}", "target")
    with c4: kpi("Growth since start", f"{growth:+.1f}%", "trend")

    st.markdown("---")

    # ---- Fuel mix ----
    df_country_fuels = panel[(panel.country == country) & (panel.fuel_code != "TOTAL")].copy()
    if "end_use_code" in df_country_fuels.columns:
        m = df_country_fuels.end_use_code == "FC_OTH_HH_E"
        if m.any():
            df_country_fuels = df_country_fuels[m]
    fuel_mix = df_country_fuels.groupby("fuel_code")["energy_tj"].sum().sort_values(ascending=False)
    fuel_total = fuel_mix.sum() if len(fuel_mix) > 0 else 0
    fuel_pct = (fuel_mix / fuel_total * 100).round(2) if fuel_total > 0 else pd.Series(dtype=float)

    df_eu_fuels = panel[panel.fuel_code != "TOTAL"].copy()
    if "end_use_code" in df_eu_fuels.columns:
        m = df_eu_fuels.end_use_code == "FC_OTH_HH_E"
        if m.any():
            df_eu_fuels = df_eu_fuels[m]
    eu_fuel_mix = df_eu_fuels.groupby("fuel_code")["energy_tj"].sum().sort_values(ascending=False)
    eu_fuel_total = eu_fuel_mix.sum() if len(eu_fuel_mix) > 0 else 0
    eu_fuel_pct = (eu_fuel_mix / eu_fuel_total * 100).round(2) if eu_fuel_total > 0 else pd.Series(dtype=float)

    # ---- Problems ----
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
        kind_map = {"High": "danger", "Medium": "warn", "Info": "info"}
        for p in problems:
            info_box(f"<b>[{p['severity']}] {p['issue']}</b><br>{p['evidence']}", kind=kind_map.get(p["severity"], "info"))

    st.markdown("---")

    # ---- Fuel mix comparison ----
    st.subheader("📊 Fuel mix — country vs Europe")
    if len(fuel_pct) == 0 and len(eu_fuel_pct) == 0:
        info_box("<b>No fuel-mix data available.</b>", kind="warn")
    else:
        all_codes = sorted(set(fuel_pct.index) | set(eu_fuel_pct.index))
        compare_df = pd.DataFrame([{
            "fuel_code": c, "country_pct": float(fuel_pct.get(c, 0.0)),
            "eu_pct": float(eu_fuel_pct.get(c, 0.0))} for c in all_codes])
        compare_df["gap"] = compare_df["country_pct"] - compare_df["eu_pct"]
        compare_df = compare_df.sort_values("country_pct", ascending=False).head(10)
        st.dataframe(compare_df.style.format({"country_pct": "{:.2f}%", "eu_pct": "{:.2f}%", "gap": "{:+.2f} pp"}),
                     use_container_width=True, hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name=country, x=compare_df.fuel_code, y=compare_df.country_pct, marker_color=T["primary"]))
        fig.add_trace(go.Bar(name="Europe average", x=compare_df.fuel_code, y=compare_df.eu_pct, marker_color=T["text_mute"]))
        fig.update_layout(barmode="group", height=420, xaxis_title="Fuel code", yaxis_title="% of fuel mix")
        st.plotly_chart(styled_plotly(fig), use_container_width=True)

    # ---- End-use comparison ----
    if "end_use_short" in panel.columns:
        st.subheader("📊 End-use mix — country vs Europe")
        enduse_source = load_csv(locate("households_by_enduse.csv"))
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
            enduse_df = pd.DataFrame([{"end_use": eu, "country_pct": float(c_pct.get(eu, 0.0)),
                                       "eu_pct": float(e_pct.get(eu, 0.0))} for eu in all_enduse])
            enduse_df["gap"] = enduse_df["country_pct"] - enduse_df["eu_pct"]
            st.dataframe(enduse_df.style.format({"country_pct": "{:.2f}%", "eu_pct": "{:.2f}%", "gap": "{:+.2f} pp"}),
                         use_container_width=True, hide_index=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(name=country, x=enduse_df.end_use, y=enduse_df.country_pct, marker_color=T["primary"]))
            fig.add_trace(go.Bar(name="Europe average", x=enduse_df.end_use, y=enduse_df.eu_pct, marker_color=T["text_mute"]))
            fig.update_layout(barmode="group", height=420, xaxis_title="End-use", yaxis_title="% of mix")
            st.plotly_chart(styled_plotly(fig), use_container_width=True)

    # ---- Trend + forecast ----
    if forecasts is not None:
        st.subheader("📈 Historical trend + forecast")
        fc = forecasts[forecasts.country == country].sort_values("year")
        if not fc.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=c_hist.year, y=c_hist.energy_tj, mode="lines+markers", name="Historical", line=dict(color=T["primary"], width=3)))
            fig.add_trace(go.Scatter(x=fc.year, y=fc.forecast_tj, mode="lines+markers", name="Forecast", line=dict(color=T["accent"], dash="dash", width=3)))
            if all(c in fc.columns for c in ["lo95_tj", "hi95_tj"]):
                fig.add_trace(go.Scatter(x=pd.concat([fc.year, fc.year[::-1]]), y=pd.concat([fc.hi95_tj, fc.lo95_tj[::-1]]),
                                         fill="toself", fillcolor="rgba(245,158,11,0.15)",
                                         line=dict(color="rgba(255,255,255,0)"), name="95% CI"))
            fig.update_layout(height=400, xaxis_title="Year", yaxis_title="Energy (TJ)", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(styled_plotly(fig), use_container_width=True)


# ============================================================
# PAGE — CLUSTERING
# ============================================================
elif page == "🎯 Clustering":
    st.header("Country Clusters")
    st.caption("KMeans segmentation of European countries by household energy profile.")

    bundle = load_clustering_bundle(str(P("clustering_bundle.joblib")))
    clusters = load_csv(locate("phase5_clusters.csv"))
    profile = load_csv(locate("phase5_cluster_profile.csv"))

    if bundle is None or clusters is None:
        info_box("<b>Clustering artifacts not found.</b> Run <b>Phase 5</b> notebook first.", kind="warn")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Countries", len(clusters), "globe")
    with c2: kpi("Clusters (k)", bundle.get("k", "—"), "target")
    with c3: kpi("Features", len(bundle.get("features", [])), "grid")
    with c4: kpi("Algorithm", "KMeans", "cog")

    st.markdown("---")
    st.subheader("PCA 2D projection")
    show_figure("phase5_pca_clusters.png")

    st.subheader("Cluster profile")
    if profile is not None:
        st.dataframe(profile, use_container_width=True)

    st.subheader("Country lookup")
    country = st.selectbox("Select a country", sorted(clusters.country.unique()), index=0)
    row = clusters[clusters.country == country].iloc[0]
    info_box(f"<b>{country}</b> belongs to <b>Cluster {row['cluster_kmeans']}</b> — "
             f"<i>{row.get('cluster_label', 'Unlabeled')}</i>")


# ============================================================
# PAGE — MODEL REPORT
# ============================================================
elif page == "📋 Model Report":
    st.header("Model Report")
    st.caption("Consolidated evaluation artifacts.")

    st.subheader("Prediction (Phase 3)")
    for nm in ("phase3_model_comparison.csv", "phase3_feature_importance.csv"):
        d = load_csv(locate(nm))
        if d is not None: st.dataframe(d, use_container_width=True)
    show_figure("phase3_feature_importance.png")

    st.subheader("Forecasting (Phase 4 + Phase 6)")
    for nm in ("phase4_backtest_results.csv", "phase4_best_model_per_country.csv",
               "phase6_model_comparison.csv", "phase6_model_by_horizon.csv"):
        d = load_csv(locate(nm))
        if d is not None: st.dataframe(d, use_container_width=True)

    st.subheader("Clustering (Phase 5)")
    d = load_csv(locate("phase5_cluster_profile.csv"))
    if d is not None: st.dataframe(d, use_container_width=True)

st.markdown("---")
st.caption("Eurostat data (CC-BY) · Built with Python & Streamlit")