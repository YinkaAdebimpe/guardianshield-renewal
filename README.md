# Engagement 1 — Customer Renewal Prediction

**Client:** GuardianShield Insurance (fictional)
**Engagement:** Customer Renewal Prediction
**Author:** Yinka Adebimpe
**Repo:** **Repo:** [guardianshield-renewal](https://github.com/YinkaAdebimpe/guardianshield-renewal)

---

## Executive Summary

GuardianShield Insurance renews **5,000 policyholders** each year. Leadership wanted to know: **which customers are most likely to lapse at renewal, and what drives that behaviour?**

This engagement delivers a **binary classification model** that predicts, at each policy anniversary, whether a customer will renew. The model achieves **ROC-AUC 0.81** and identifies four behavioural triggers that a retention team can act on before the renewal date.

**Headline findings:**

| Metric | Value |
|--------|-------|
| Renewal decisions modelled | 19,161 |
| Target balance | 54.6% renew / 45.4% lapse |
| Best model ROC-AUC | **0.807** (Random Forest) |
| Top churn driver | **Missed payments** (hazard signal) |
| Second churn driver | **Last year's renewal** (stickiness) |

**Four behavioural signals dominate:**
1. **Last year's renewal** — customers who renewed last year are far more likely to renew again
2. **Missed payments** — the strongest *negative* signal; a leading indicator of disengagement
3. **Premium increase size** — customers facing >10% increases are significantly more likely to lapse
4. **Cumulative loyalty** — customers with 3+ prior renewals are markedly stickier

**Recommended actions:**

1. **Trigger retention outreach on missed payments** — the strongest controllable churn signal
2. **Pre-empt large premium increases** with a retention offer before the renewal date
3. **Recognise long-tenure customers** — protect the relationship, not through discounting but through recognition
4. **Downweight demographics** — age, region, and excess add almost nothing to prediction

---

## Method

### Data

- **Source:** Synthetic renewal data (fully reproducible via `src/generate_renewal_data.py`)
- **Scale:** 5,000 customers, 19,207 renewal decisions, 2018–2025
- **Unit:** One row per renewal decision (a customer with 5 years of tenure contributes 5 rows)
- **Segments:** Three latent segments (loyal / standard / price-sensitive) with distinct renewal propensities

### Approach

1. **EDA** (`notebooks/01_eda.ipynb`) — distribution analysis, feature-target relationships
2. **Feature engineering** (`notebooks/02_feature_engineering.ipynb`) — categorical encoding, derived features
3. **Modelling** (`notebooks/03_modeling.ipynb`) — four classifiers benchmarked
4. **Iteration** — initial features produced ROC-AUC ~0.70; adding four behavioural features raised it to **0.81**

### Models compared

| Model | ROC-AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| **Random Forest** | **0.807** | 0.697 | 0.745 | 0.720 |
| Logistic Regression | 0.806 | 0.684 | 0.748 | 0.715 |
| Decision Tree | 0.798 | 0.681 | 0.766 | 0.721 |
| XGBoost | 0.796 | 0.693 | 0.709 | 0.701 |

All four models cluster tightly in the 0.80–0.81 band — indicating the signal is largely linear.

---

## Repository structure

```text
.
├── data/
│   ├── raw/                          # Generated CSVs (gitignored)
│   └── processed/                    # Committed summary CSVs for GitHub review
│       ├── features_v1.csv
│       └── predictions_v1.csv
├── images/                           # Analysis and model plots
├── models/                           # Trained model artifacts (.pkl)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── reports/                          # Executive summary, technical report
├── sql/                              # SQL scripts
├── src/
│   ├── config.py
│   ├── generate_renewal_data.py
│   └── load_to_postgres.py
├── .env.example                      # Template — copy to .env and fill in
├── .gitignore
├── README.md
└── requirements.txt
```

---

## How to reproduce

### Prerequisites

- Python 3.12+
- PostgreSQL 14+ (running on `localhost:5432`)
- A database named `guardianshield_db`

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/YinkaAdebimpe/guardianshield-renewal.git
cd Engagement-1-Customer-Renewal

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure database credentials
cp .env.example .env
# Edit .env with your Postgres password (DB_SCHEMA=retention)
```

### Run the pipeline

```bash
# Generate synthetic renewal data (~5,000 customers, ~19,000 decisions)
cd src
python generate_renewal_data.py

# Load into PostgreSQL (creates schema `retention`)
python load_to_postgres.py

# Launch Jupyter and run the notebooks in order
cd ..
jupyter lab
```

Then open the notebooks in `notebooks/` and run them in order:

1. `01_eda.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_modeling.ipynb`

---

## Results

### Feature importance (top 5)

| Rank | Feature | Importance | Business signal |
|------|---------|-----------|-----------------|
| 1 | `last_year_renewed` | 0.290 | Past renewal predicts future renewal |
| 2 | `missed_payments` | 0.141 | Disengagement leading indicator |
| 3 | `annual_premium` | 0.098 | Higher premium → stickier customer |
| 4 | `premium_change_pct` | 0.080 | Large price rises trigger churn |
| 5 | `previous_renewals_count` | 0.054 | Cumulative loyalty |

### Business impact framework

The model flags **high-risk lapses** with ROC-AUC 0.807. If the retention team targets the top 20% highest-risk customers:

- Approximately **X% of actual lapses** are captured (see confusion matrix)
- Each saved renewal is worth the customer's expected CLV (see Engagement 2)
- A conservative retention offer costing £20 with 15% success rate has a strong positive ROI

*(Exact figures depend on the retention programme's cost-per-success; see Technical Report for sensitivity analysis.)*

### Visualisations

- `images/target_distribution.png` — target balance and segment renewal rates
- `images/feature_distributions.png` — numeric feature histograms
- `images/renewal_rate_by_feature.png` — renewal rate by tenure, frequency, claims, policy type
- `images/correlation_matrix.png` — numeric correlation structure
- `images/model_roc_curves.png` — ROC curves for all four models
- `images/threshold_tuning.png` — precision/recall/F1 vs threshold
- `images/confusion_matrix.png` — confusion matrix at optimal threshold
- `images/feature_importance.png` — top 15 feature importances

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.12 |
| Data manipulation | pandas, numpy |
| Machine learning | scikit-learn, XGBoost |
| Database | PostgreSQL 18, SQLAlchemy, psycopg2 |
| Visualisation | matplotlib, seaborn |
| Notebooks | JupyterLab |
| Version control | Git, GitHub |

---

## Limitations

1. **Synthetic data** — reproduces realistic renewal behaviour but is not real client data. Feature effect sizes were tuned to reflect UK insurance benchmarks.
2. **Segment label excluded** — the latent segment label was not used as a feature, to avoid leakage. The model must infer segment-like behaviour from observable signals.
3. **No temporal validation** — the train/test split is random, not time-based. A production model should be validated on a holdout period (e.g. train up to 2023, test on 2024–25).
4. **No hyperparameter tuning** — model settings are sensible defaults. A production version would tune via cross-validation.
5. **Threshold assumption** — the optimal threshold maximises F1. In production, the threshold should be chosen to align with the cost of a false positive (wasted retention offer) vs false negative (lost customer).

---

## Author

**Yinka Adebimpe**
Data Scientist | Customer Analytics & Predictive Modelling

I build end-to-end data science solutions for customer-facing businesses — from data engineering and feature design to survival analysis, machine learning, and business-ready reporting.

- GitHub: [@YinkaAdebimpe](https://github.com/YinkaAdebimpe)
- Email: adebimpey@gmail.com

---

*Part of the OkYeahInsight Analytics portfolio — a three-engagement consulting series for GuardianShield Insurance.*

**Note on shared environment:** All three GuardianShield engagements share the Python virtual environment at `C:\venvs\guardianshield-clv`. This keeps dependencies consistent across engagements and avoids duplicating ~500 MB of packages per project.