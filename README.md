# ⚡ European Household Energy Intelligence System (2010–2024)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E)
![Status](https://img.shields.io/badge/Status-Complete-success)

An end-to-end Data Science and Machine Learning project exploring how European households consumed energy over 15 years. This repository covers the full lifecycle: **Data Analysis, Cleaning, EDA, Advanced ML Modeling, Time-Series Forecasting, Clustering, and a Live Interactive Dashboard**.

![Trend](reports/figures/fig01_total_trend.png)

---

## 📊 Project Overview

**Core Question:** How has household energy consumption in Europe changed between 2010 and 2024, where does it come from, and how can we predict its future?

**Scope:**
- 30+ European countries
- 15 years (2010–2024)
- 10+ fuel types and multiple end-use categories
- Source: Eurostat household energy final consumption

**Deliverables:**
1. Reproducible Jupyter Notebooks (Data Analysis to Advanced ML).
2. Exportable ML Models (`.joblib`).
3. A fully interactive Streamlit Dashboard.

---

## 📂 Repository Structure

```text
European_Household_Energy_Project/
│
├── data/
│   ├── raw/                   # Source Eurostat CSV
│   └── processed/             # Cleaned datasets (panel_level.csv, etc.)
│
├── models/
│   ├── advanced_prediction_bundle.joblib
│   ├── forecasting_bundle.joblib
│   └── clustering_bundle.joblib
│
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Exploratory_Data_Analysis.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Business_Analysis.ipynb
│   ├── 06_Visualizations.ipynb
│   ├── Phase 5_Country_Clustering.ipynb
│   ├── 06_Advanced_Forecasting.ipynb
│   └── Energy_Prediction_Advanced_ML.ipynb
│
├── outputs/
│   ├── figures/               # PNG charts for reports
│   ├── predictions/           # CSV forecasts (2025-2027)
│   └── evaluation_reports/    # Model comparison metrics
│
├── app.py                     # Streamlit Dashboard
├── requirements.txt           # Dependencies
└── README.md