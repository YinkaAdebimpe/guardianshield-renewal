# Engagement 1 — Customer Renewal Prediction

**Client:** GuardianShield Insurance (fictional)  
**Engagement:** Customer Renewal Prediction  
**Author:** Yinka Adebimpe  
**Repo:** [guardianshield-renewal](https://github.com/YinkaAdebimpe/guardianshield-renewal)

---

## Business Context & Impact

**The Business Problem:**
GuardianShield Insurance renews 5,000 policyholders each year. Customer churn at renewal was eroding recurring revenue, and the business lacked a systematic way to identify which customers were at risk of lapsing.

**The Data Solution:**
I built a binary classification model that predicts, at each policy anniversary, whether a customer will renew. The model achieves **ROC-AUC 0.81** and identifies four behavioural triggers (missed payments, last year's renewal, premium increases, cumulative loyalty) that a retention team can act on *before* the renewal date.

**The Business Impact:**
By identifying high-risk segments and their key churn drivers, the model allows targeted retention campaigns rather than blanket discounts. At a projected portfolio CLV of £3.33M, a conservative 5% retention lift translates to **~£166,000 in protected annual revenue**.

---

## Technical Summary

| Metric | Value |
|--------|-------|
| Renewal decisions modelled | 19,161 |
| Target balance | 54.6% renew / 45.4% lapse |
| Best model ROC-AUC | **0.807** (Random Forest) |
| Top churn driver | Missed payments (hazard signal) |
| Second churn driver | Last year's renewal (stickiness) |

### Four behavioural signals dominate:
1. **Last year's renewal** — past behaviour predicts future renewal.
2. **Missed payments** — the strongest *negative* signal; a leading indicator of disengagement.
3. **Premium increase size** — customers facing >10% increases are significantly more likely to lapse.
4. **Cumulative loyalty** — customers with 3+ prior renewals are markedly stickier.

### Recommended Actions:
- **Trigger retention outreach on missed payments** — the strongest controllable churn signal.
- **Pre-empt large premium increases** with a retention offer before the renewal date.
- **Recognise long-tenure customers** — protect the relationship through recognition, not just discounting.
- **Downweight demographics** — age, region, and excess add almost nothing to prediction.

---

## Methodology

### Data
- **Source:** Synthetic renewal data (fully reproducible via `src/generate_renewal_data.py`)
- **Scale:** 5,000 customers, 19,207 renewal decisions, 2018–2025
- **Unit:** One row per renewal decision (a customer with 5 years of tenure contributes 5 rows)
- **Segments:** Three latent segments (loyal / standard / price-sensitive) with distinct renewal propensities.

### Approach
1. **EDA** (`notebooks/01_eda.ipynb`) — distribution analysis, feature-target relationships.
2. **Feature engineering** (`notebooks/02_feature_engineering.ipynb`) — categorical encoding, derived features.
3. **Modelling** (`notebooks/03_modeling.ipynb`) — four classifiers benchmarked.
4. **Iteration** — initial features produced ROC-AUC ~0.70; adding four behavioural features raised it to **0.81**.

### Models compared
| Model | ROC-AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| **Random Forest** | **0.807** | 0.697 | 0.745 | 0.720 |
| Logistic Regression | 0.806 | 0.684 | 0.748 | 0.715 |
| Decision Tree | 0.798 | 0.681 | 0.766 | 0.721 |
| XGBoost | 0.796 | 0.693 | 0.709 | 0.701 |

*All four models cluster tightly in the 0.80–0.81 band — indicating the signal is largely linear.*

---

## Repository Structure

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

## How to Reproduce

### Prerequisites
- Python 3.12+
- PostgreSQL 14+ (running on `localhost:5432`)
- A database named `guardianshield_db`

### Setup
```bash
# 1. Clone the repo
git clone https://github.com/YinkaAdebimpe/guardianshield-renewal.git
cd guardianshield-renewal

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure database
cp .env.example .env
# Edit .env with Postgres credentials (DB_SCHEMA=retention)
```

### Run the pipeline
```bash
cd src
python generate_renewal_data.py
python load_to_postgres.py
cd ..
jupyter lab
```

*Prepared as part of the OkYeahInsight Analytics portfolio — a three-engagement consulting series for GuardianShield Insurance.*
