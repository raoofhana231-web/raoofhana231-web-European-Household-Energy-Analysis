# European Household Energy Project

# 🏠 European Household Energy Analysis (2010–2024)

A professional data analysis project exploring how European households consumed energy over 15 years — by country, fuel type, and end-use — using **Eurostat** data.

![Trend](reports/figures/fig01_total_trend.png)

---

## 1. Project Overview

**Question:** How has household energy consumption in Europe changed between 2010 and 2024 — and where does it come from?

**Scope:**
- 30+ European countries
- 15 years (2010–2024)
- 10+ fuel types and 3 end-use categories
- Source: Eurostat household energy final consumption

**Deliverable:** A reproducible Python notebook pipeline (understanding → cleaning → EDA → features → business analysis → visuals) that produces decision-grade insights.

---

## 2. Dataset Description

| Field | Description |
|---|---|
| `country` | Country name |
| `is_country_group` | True for EU-level aggregates (excluded from country analysis) |
| `end_use_code` | Household end-use category (Total / Cooking / Lighting / Other) |
| `fuel_code` | Fuel type (Electricity, Gas, Oil, Renewables, etc.) |
| `year` | Reporting year |
| `energy_tj` | Energy consumption in Terajoules |
| `flag` | Quality metadata (estimated, provisional, etc.) |

Full data dictionary available in `01_Data_Understanding.ipynb`.

---

## 3. Data Understanding

- **Rows:** ~ thousands of country × year × fuel × end-use records
- **Numerical:** `year`, `energy_tj`
- **Categorical:** country, fuel, end-use, flags
- **Temporal:** `date` (converted to datetime in cleaning)

Key structural facts:
- **Country groups** and **real countries** coexist → filtered to avoid double counting.
- **Aggregated fuels** (`RA000`, `O4000`) exist alongside their sub-components.
- **Zero values** can mean *no consumption* or *non-reported data*.

---

## 4. Data Quality Assessment

| Dimension | Finding |
|---|---|
| Missing values | Concentrated in `flag` / `flag_meaning` — **not** in `energy_tj` |
| Duplicates | None on the analytical key (country, year, fuel, end-use) |
| Negative values | None |
| Zero values | Classified as `valid_zero` vs `all_zero_series` |
| Coverage | Some countries have < 15 years of data (documented) |
| Total vs sum-of-fuels | Consistent within ±5% for most countries |

---

## 5. Data Cleaning Steps

1. Normalize string columns (strip whitespace, empty → NaN)
2. Convert `date` → `datetime` with year-consistency check
3. Drop exact duplicates
4. **Separate country groups** from individual countries
5. **Tag fuel hierarchy**: `aggregate`, `primary`, `sub_fuel`
6. **Classify zeros**: `valid_zero` vs `all_zero_series`
7. Audit year coverage per country
8. Cross-check `TOTAL` against sum of primary fuels
9. **Preserve flags** for downstream transparency
10. Export three clean analytical views + cleaning log

---

## 6. Exploratory Data Analysis

**Aggregate:** Total household energy declined ~11% from 2010 to 2024, with a temporary COVID-19 bump.

**Countries:** The top 5 (Germany, France, Italy, UK, Spain) hold ≈ 65% of total; concentration is stable over time.

**Fuels:** Electricity + Natural gas ≈ 60% of the mix. Renewables (mostly solid biofuels) hold ~20% but are declining in absolute terms.

**End-uses:** "Other" (mostly space/water heating) ≈ 70%; Cooking ≈ 25% (high because of Balkan biomass); Lighting & appliances dropped sharply after 2018 (LED + efficiency).

---

## 7. Key Visualizations

| Chart | Message |
|---|---|
| ![Total](reports/figures/fig01_total_trend.png) | **Total declined ≈ 11% with a COVID bump** |
| ![Top 10](reports/figures/fig02_top10.png) | **Top 5 = 65% of the total** |
| ![Mix](reports/figures/fig03_fuel_mix.png) | **Electricity + gas dominate** |
| ![Renew](reports/figures/fig04_renew_fossil.png) | **Renewables gaining, but slowly** |
| ![End-use](reports/figures/fig05_enduse.png) | **Heating dominates; lighting collapsed** |
| ![Mix shift](reports/figures/fig06_mix_shift.png) | **Some countries restructured deeply** |
| ![Compare](reports/figures/fig07_country_compare.png) | **Country head-to-head comparison** |

---

## 8. Major Findings

1. **Household energy is falling, but slowly (~ −0.8%/year on average).**
2. **Concentration is structural** — the same 5 countries lead every year.
3. **The fuel mix is slowly shifting** — electricity holds, gas softens, renewables grow slowly.
4. **Heating is the dominant end-use** — and also the hardest to decarbonize.
5. **Appliance efficiency already delivered** — lighting & appliances dropped from ~15% to ~5%.
6. **Country-level mix shifts vary widely** — some countries halved their coal/gas share; others are locked in.

---

## 9. Business Insights

| Audience | Insight |
|---|---|
| **Policy makers** | 65% of the impact sits in 5 countries → target them first. |
| **Utility strategists** | Gas-to-electric conversion is the largest single lever. |
| **Equipment manufacturers** | Heat-pump demand will outpace appliance demand. |
| **ESG investors** | Renewable share growth is real but slow — carbon risk on gas-heavy utilities remains high. |
| **Building owners** | Retrofits (insulation + heat pumps) yield more carbon reduction than appliance upgrades. |

---

## 10. Recommendations

1. **Prioritize the top 5 markets** — Germany, France, Italy, UK, Spain — for any household energy program.
2. **Focus on heating retrofits** — that's where 70% of household energy sits.
3. **Accelerate heat-pump adoption** — the single largest lever for gas displacement.
4. **Don't rely on small renewable niches** — solar thermal and heat pumps need scale-up.
5. **Address locked-in gas economies** — structural programs needed where the mix hasn't moved.
6. **Maintain data quality flags** — acknowledge estimation uncertainty in public reporting.

---

## 11. Limitations

- **No population, floor-area, or weather normalization** — cannot claim efficiency improvements directly.
- **"Other end uses" hides heating types** — finer disaggregation would sharpen insights.
- **Some countries have short or noisy series** — cross-country comparison requires caution.
- **2024 data may be provisional** — flagged in the source.
- **No income, behavior, or price data** — cannot model demand elasticity.

---

## 12. Conclusion

Europe's household energy transition is **real but too slow**. Demand is falling, renewables are rising, but the structural lock-in of gas heating in the largest economies means the continent is unlikely to hit its 2030 residential targets without **aggressive policy on building retrofits and heat-pump deployment**. The data shows clearly where the impact lives — in heating, in the top 5 countries, and in the gas-to-electric switch.

---

## 📁 Repository Structure

```
european-household-energy/
├── data/
│   ├── raw/
│   └── processed/
│       ├── households_country_total.csv
│       ├── households_by_fuel.csv
│       ├── households_by_enduse.csv
│       ├── data_cleaning_log.csv
│       └── features/
│           ├── features_country_year.csv
│           ├── features_fuel_level.csv
│           └── features_enduse_level.csv
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Exploratory_Data_Analysis.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Business_Analysis.ipynb
│   └── 06_Visualizations.ipynb
├── reports/
│   └── figures/  (7 PNG files)
├── README.md
└── requirements.txt
```

## 🚀 How to reproduce

```bash
git clone <repo>
cd european-household-energy
pip install -r requirements.txt
jupyter lab notebooks/
```

Run notebooks in order: **01 → 02 → 03 → 04 → 05 → 06**.

---

**Author:** *<Hana>*
**Data source:** Eurostat — Final energy consumption in households
**License:** For educational and portfolio purposes