"""
improve_pipeline.py
===================
Run from the project root:   python scripts/improve_pipeline.py

What it does
  1. PREDICTION  : fair baselines (persistence, moving-average), fixed Linear/Ridge
                   (log-target, no more 1000%+ WAPE), HistGB, and exports
                   models/advanced_prediction_bundle.joblib for the dashboard.
  2. FORECASTING : rolling-origin validation of 6 candidates, per-country selection,
                   empirical confidence intervals, 2025-2029 forecasts.
  3. CLUSTERING  : silhouette / Calinski-Harabasz / Davies-Bouldin for k = 2..6.
"""
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "outputs" / "evaluation_reports"
FIGS = ROOT / "outputs" / "figures"
ADV = ROOT / "outputs" / "advanced_forecasting"
for d in (MODELS, REPORTS, FIGS, ADV):
    d.mkdir(parents=True, exist_ok=True)


def wape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / np.abs(y).sum() * 100)


def _expm1_clip(z):
    return np.expm1(np.clip(z, 0, 20))


# =====================================================================
# 1. PREDICTION
# =====================================================================
print("== 1. Prediction ==")
panel = pd.read_csv(PROC / "panel_level.csv")
KEY = ["country", "end_use_code", "fuel_code"]
panel = panel.sort_values(KEY + ["year"]).reset_index(drop=True)
grp = panel.groupby(KEY)["energy_tj"]
for k in (1, 2, 3):
    panel[f"lag_{k}"] = grp.shift(k)
LAGS = ["lag_1", "lag_2", "lag_3"]
panel["rolling_mean_3"] = panel[LAGS].mean(axis=1)
panel["rolling_std_3"] = panel[LAGS].std(axis=1, ddof=0)
MIN_YEAR = int(panel.year.min())
panel["year_index"] = panel.year - MIN_YEAR
panel["year_sq"] = panel.year_index ** 2
data = panel.dropna(subset=LAGS).copy()

CAT = ["country", "end_use_code", "fuel_code"]
NUM = ["year", "year_index", "year_sq"] + LAGS + ["rolling_mean_3", "rolling_std_3"]
FEATURES = CAT + NUM

train = data[data.year <= 2019]
val = data[(data.year >= 2020) & (data.year <= 2022)]
test = data[data.year >= 2023]
print(f"rows  train={len(train):,}  val={len(val):,}  test={len(test):,}")


def make_hgb():
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)],
        remainder="passthrough",
    )
    reg = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
        categorical_features=[0, 1, 2], random_state=42,
    )
    return TransformedTargetRegressor(
        regressor=Pipeline([("pre", pre), ("model", reg)]),
        func=np.log1p, inverse_func=np.expm1,
    )


def make_linear(ridge):
    log_cols = LAGS + ["rolling_mean_3", "rolling_std_3"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("log", FunctionTransformer(np.log1p), log_cols),
        ("yr", StandardScaler(), ["year_index"]),
    ])
    reg = Ridge(alpha=10.0) if ridge else LinearRegression()
    return TransformedTargetRegressor(
        regressor=Pipeline([("pre", pre), ("model", reg)]),
        func=np.log1p, inverse_func=_expm1_clip,
    )


def fit_predict(factory, tr, te):
    m = factory()
    m.fit(tr[FEATURES], tr["energy_tj"])
    return np.clip(m.predict(te[FEATURES]), 0, None)


candidates = {
    "Persistence (last year)": lambda tr, te: te["lag_1"].values,
    "MovingAverage3": lambda tr, te: te["rolling_mean_3"].values,
    "Linear (log, fixed)": lambda tr, te: fit_predict(lambda: make_linear(False), tr, te),
    "Ridge (log, fixed)": lambda tr, te: fit_predict(lambda: make_linear(True), tr, te),
    "HistGB (log target)": lambda tr, te: fit_predict(make_hgb, tr, te),
}
rows, full_rows = [], []
for name, fn in candidates.items():
    r = {"model": name}
    for split_name, df in (("val", val), ("test", test)):
        p = fn(train, df)
        r[split_name] = round(wape(df.energy_tj, p), 3)
        y = df.energy_tj.values
        full_rows.append({
            "model": name, "split": split_name,
            "MAE": float(np.abs(y - p).mean()),
            "RMSE": float(np.sqrt(((y - p) ** 2).mean())),
            "R2": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "WAPE_%": r[split_name],
        })
    rows.append(r)
    print(f"  {name:26s} val WAPE={r['val']:7.2f}%  test WAPE={r['test']:7.2f}%")

comp = pd.DataFrame(rows).sort_values("test").reset_index(drop=True)
comp.to_csv(REPORTS / "phase3_model_comparison.csv", index=False)      # best model first
pd.DataFrame(full_rows).to_csv(REPORTS / "phase3_full_metrics.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(comp))
ax.bar(x - 0.2, comp["val"], 0.4, label="Validation (2020-22)", color="#818cf8")
ax.bar(x + 0.2, comp["test"], 0.4, label="Test (2023-24)", color="#10b981")
ax.set_xticks(x)
ax.set_xticklabels(comp.model, rotation=20, ha="right")
ax.set_ylabel("WAPE %")
ax.set_title("Prediction models vs. fair baselines")
ax.legend()
plt.tight_layout()
plt.savefig(FIGS / "phase3_baseline_comparison.png", dpi=140)
plt.close()

# ---- deploy bundle: HistGB refit on ALL rows --------------------------------
hgb_final = make_hgb()
hgb_final.fit(data[FEATURES], data["energy_tj"])
joblib.dump({
    "model": hgb_final, "features": FEATURES, "min_year": MIN_YEAR,
    "model_name": "HistGB (log target)", "sklearn_version": sklearn.__version__,
    "trained_on": f"{int(data.year.min())}-{int(data.year.max())}",
}, MODELS / "advanced_prediction_bundle.joblib")
print("  saved models/advanced_prediction_bundle.joblib")

# =====================================================================
# 2. FORECASTING
# =====================================================================
print("== 2. Forecasting ==")
ct = pd.read_csv(PROC / "country_total.csv")
ct = ct.groupby(["country", "year"], as_index=False)["energy_tj"].sum().sort_values(["country", "year"])
LAST_YEAR = int(ct.year.max())
HMAX = 5


def f_naive(y, h):
    return np.repeat(y[-1], h)


def f_ma3(y, h):
    return np.repeat(y[-3:].mean(), h)


def f_drift5(y, h):
    d = np.diff(y[-6:]).mean()
    steps = np.cumsum(0.8 ** np.arange(1, h + 1))          # damped drift
    return y[-1] + d * steps


def f_linear8(y, h):
    yy = y[-8:]
    x = np.arange(len(yy))
    s, i = np.polyfit(x, yy, 1)
    return i + s * (len(yy) - 1 + 0.5 * np.arange(1, h + 1))   # half-slope: conservative


def f_holt(y, h):
    try:
        m = ExponentialSmoothing(y, trend="add", damped_trend=True,
                                 initialization_method="estimated").fit()
        return np.asarray(m.forecast(h))
    except Exception:
        return f_naive(y, h)


def f_ensemble(y, h):
    return np.mean([f_naive(y, h), f_drift5(y, h), f_holt(y, h), f_linear8(y, h)], axis=0)


MODELS_TS = {"Naive": f_naive, "MovingAverage3": f_ma3, "DampedDrift": f_drift5,
             "LinearTrend8": f_linear8, "HoltDamped": f_holt, "Ensemble": f_ensemble}

records = []
for country, g in ct.groupby("country"):
    s = g.set_index("year").energy_tj
    for origin in range(2014, LAST_YEAR):
        tr = s[s.index <= origin]
        if len(tr) < 6:
            continue
        hmax = min(HMAX, LAST_YEAR - origin)
        actual = s.reindex(range(origin + 1, origin + hmax + 1))
        for name, fn in MODELS_TS.items():
            fc_ = np.asarray(fn(tr.values.astype(float), hmax), float)
            for h in range(1, hmax + 1):
                if not np.isnan(actual.iloc[h - 1]):
                    records.append((country, name, origin, h, actual.iloc[h - 1], fc_[h - 1]))
bt = pd.DataFrame(records, columns=["country", "model", "origin", "h", "actual", "forecast"])
bt["abs_err"] = (bt.actual - bt.forecast).abs()
bt["rel_err"] = (bt.actual - bt.forecast) / bt.forecast.replace(0, np.nan)


def wape_df(d):
    return d.abs_err.sum() / d.actual.abs().sum() * 100


comp_ts = (bt.groupby("model").apply(wape_df).rename("WAPE_%").reset_index()
           .sort_values("WAPE_%").reset_index(drop=True))
comp_ts["WAPE_%"] = comp_ts["WAPE_%"].round(3)
comp_ts.to_csv(REPORTS / "phase6_model_comparison.csv", index=False)

by_h = bt.groupby(["model", "h"]).apply(wape_df).unstack("h").round(3)
by_h.columns = [f"h{c}_WAPE_%" for c in by_h.columns]
by_h.reset_index().to_csv(REPORTS / "phase6_model_by_horizon.csv", index=False)
print(comp_ts.to_string(index=False))

per_cm = bt.groupby(["country", "model"]).apply(wape_df).rename("wape").reset_index()
per_cm.round(3).to_csv(REPORTS / "phase4_backtest_results.csv", index=False)

global_best = comp_ts.iloc[0]["model"]
choice = []
for country, g in per_cm.groupby("country"):
    g = g.set_index("model").wape
    best_local = g.idxmin()
    # per-country override only if >=20% better than the global winner (avoids over-fitting)
    sel = best_local if g[best_local] <= 0.8 * g[global_best] else global_best
    choice.append({"country": country, "best_model": sel, "best_wape": round(g[sel], 3)})
choice = pd.DataFrame(choice)
choice.to_csv(REPORTS / "phase4_best_model_per_country.csv", index=False)
print(f"  global best: {global_best}; per-country overrides: {(choice.best_model != global_best).sum()}")

# empirical error spread per horizon, per selected model, pooled across countries
sd_h = {}
for model in choice.best_model.unique():
    d = bt[bt.model == model]
    sd = d.groupby("h").rel_err.std()
    cnt = d.groupby("h").size()
    ok = [h for h in sd.index if cnt[h] >= 15 and not np.isnan(sd[h])]
    base_h = max(ok)
    for h in range(1, HMAX + 1):
        sd_h[(model, h)] = float(sd[h]) if h in ok else float(sd[base_h] * np.sqrt(h / base_h))
    for h in range(2, HMAX + 1):                       # enforce monotone growth
        sd_h[(model, h)] = max(sd_h[(model, h)], sd_h[(model, h - 1)])

fc_rows = []
for _, r in choice.iterrows():
    y = ct[ct.country == r.country].sort_values("year").energy_tj.values.astype(float)
    f = np.clip(np.asarray(MODELS_TS[r.best_model](y, HMAX), float), 0, None)
    for h in range(1, HMAX + 1):
        sd = sd_h[(r.best_model, h)]
        fc_rows.append({
            "country": r.country, "year": LAST_YEAR + h, "forecast_tj": f[h - 1], "model": r.best_model,
            "lo80_tj": max(0, f[h - 1] * (1 - 1.2816 * sd)), "hi80_tj": f[h - 1] * (1 + 1.2816 * sd),
            "lo95_tj": max(0, f[h - 1] * (1 - 1.96 * sd)), "hi95_tj": f[h - 1] * (1 + 1.96 * sd),
        })
fc = pd.DataFrame(fc_rows)
fc.to_csv(ADV / "country_forecasts_2025_2029.csv", index=False)
fc.pivot(index="country", columns="year", values="forecast_tj").reset_index().to_csv(
    ADV / "country_forecasts_pivot_2025_2029.csv", index=False)
eu = fc.groupby("year")[["forecast_tj", "lo80_tj", "hi80_tj", "lo95_tj", "hi95_tj"]].sum().reset_index()
eu.to_csv(ADV / "europe_forecast_2025_2029.csv", index=False)
pd.DataFrame([{"selected_model": global_best, "forecast_horizon": HMAX, "last_observed_year": LAST_YEAR,
               "holdout_years": "rolling origins 2014-2023",
               "selection_basis": "Pooled rolling-origin WAPE; per-country override only if >=20% better"}]
             ).to_csv(ADV / "model_selection_summary.csv", index=False)

top6 = ct.groupby("country").energy_tj.mean().nlargest(6).index
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, c in zip(axes.ravel(), top6):
    h = ct[ct.country == c]
    f = fc[fc.country == c]
    ax.plot(h.year, h.energy_tj, color="#4f46e5", marker="o", ms=3, label="Historical")
    ax.plot(f.year, f.forecast_tj, color="#10b981", ls="--", marker="D", ms=4, label="Forecast")
    ax.fill_between(f.year, f.lo95_tj, f.hi95_tj, color="#f59e0b", alpha=.2, label="95% CI")
    ax.fill_between(f.year, f.lo80_tj, f.hi80_tj, color="#f59e0b", alpha=.35, label="80% CI")
    ax.set_title(f"{c} ({f.model.iloc[0]})", fontsize=10)
axes[0, 0].legend(fontsize=8)
plt.suptitle("2025-2029 forecasts with empirical confidence intervals")
plt.tight_layout()
plt.savefig(FIGS / "phase6_forecasts_top6.png", dpi=140)
plt.close()

# =====================================================================
# 3. CLUSTERING - choice of k
# =====================================================================
print("== 3. Clustering k-selection ==")
prof = pd.read_csv(PROC / "country_profile.csv").dropna()
bundle = joblib.load(MODELS / "clustering_bundle.joblib")
feats = bundle["features"]
Xs = StandardScaler().fit_transform(prof[feats])
krows = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=30, random_state=42).fit(Xs)
    krows.append({"k": k, "silhouette": round(silhouette_score(Xs, km.labels_), 4),
                  "calinski_harabasz": round(calinski_harabasz_score(Xs, km.labels_), 2),
                  "davies_bouldin": round(davies_bouldin_score(Xs, km.labels_), 4),
                  "smallest_cluster": int(np.bincount(km.labels_).min())})
ks = pd.DataFrame(krows)
ks.to_csv(REPORTS / "phase5_k_selection.csv", index=False)
print(ks.to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(ks.k, ks.silhouette, marker="o", color="#4f46e5")
ax[0].set_title("Silhouette (higher = better)")
ax[1].plot(ks.k, ks.davies_bouldin, marker="o", color="#ef4444")
ax[1].set_title("Davies-Bouldin (lower = better)")
for a in ax:
    a.set_xlabel("k")
plt.tight_layout()
plt.savefig(FIGS / "phase5_k_selection.png", dpi=140)
plt.close()
print("done")
