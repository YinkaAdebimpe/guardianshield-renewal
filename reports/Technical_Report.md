# Customer Renewal Prediction — Technical Report

**Client:** GuardianShield Insurance *(fictional)*  
**Engagement:** Customer Renewal Prediction  
**Prepared by:** Yinka Adebimpe, Data Scientist | Customer Analytics & Predictive Modelling  
**Date:** October 2026

---

## 1. Introduction & Scope

### 1.1 Objective
GuardianShield Insurance required a predictive model of **renewal behaviour**. The model needed to:
1. Predict, at each policy anniversary, whether a customer will renew.
2. Identify the behavioural drivers of lapse.
3. Produce calibrated probabilities to support retention prioritisation.
4. Translate model performance into a tangible business case for retention spend.
5. Be fully reproducible from source data to final model artifact.

### 1.2 Scope
The engagement covers **5,000 policyholders** and **19,207 renewal decisions** spanning January 2018 – December 2025. Classification is at the **renewal decision level** — one row per customer per policy year.

---

## 2. Data

### 2.1 Source
The engagement uses **fully synthetic renewal data** generated from a reproducible Python pipeline (`src/generate_renewal_data.py`). Real client data was not available; synthetic data was used to demonstrate methodology on realistic renewal behaviour.

### 2.2 Scale and dimensions
| Property | Value |
|----------|-------|
| Customers | 5,000 |
| Renewal decisions | 19,207 |
| Observation window | January 2018 – December 2025 (8 years) |
| Mean decisions per customer | 3.84 |
| Target balance | 54.6% renew / 45.4% lapse |

### 2.3 Data Quality Checks
| Check | Result |
|-------|--------|
| All customers have ≥1 renewal decision | ✅ |
| No missing values in features | ✅ |
| All features numeric after encoding | ✅ |

---

## 3. Methodology

### 3.1 Feature Engineering
**Dropped columns:** `customer_id`, `policy_id`, `decision_date`, `policy_year`, `segment` (leakage), `region` (high cardinality).

**Derived features:** `tenure_years`, `claims_per_year`, `high_premium_change`, `had_claim_last_year`, `tenure_x_no_claim`.

**Encoded categoricals:** `payment_frequency`, `policy_type`, `acquisition_channel`, `last_year_renewed`.

**Final feature matrix:** 19,161 rows × 26 features (25 predictors + 1 target).

### 3.2 Train/Test Split
`train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)`
- Train: 15,328 rows | Test: 3,833 rows
- Stratified to preserve the 54.6/45.4 class balance.

### 3.3 Models Benchmarked
Four models were evaluated with `class_weight="balanced"` where applicable:
1. **Logistic Regression** (with StandardScaler)
2. **Decision Tree** (max_depth=6, min_samples_leaf=20)
3. **Random Forest** (n_estimators=300, max_depth=10)
4. **XGBoost** (n_estimators=300, max_depth=6)

### 3.4 Evaluation Metrics
| Metric | Purpose |
|--------|---------|
| Accuracy | Overall correctness |
| Precision | Of predicted renewals, how many actually renewed |
| Recall | Of actual renewals, how many were correctly identified |
| F1 | Harmonic mean of precision and recall |
| ROC-AUC | Ranking quality across all thresholds |

---

## 4. Results

### 4.1 Model comparison
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| **Random Forest** | **0.737** | **0.697** | 0.745 | **0.720** | **0.807** |
| Logistic Regression | 0.729 | 0.684 | 0.748 | 0.715 | 0.806 |
| Decision Tree | 0.731 | 0.681 | 0.766 | 0.721 | 0.798 |
| XGBoost | 0.725 | 0.693 | 0.709 | 0.701 | 0.796 |

**Observation:** All four models fall in a **tight 0.796–0.807 ROC-AUC band**. The signal is largely **linear**; no benefit from complex feature interactions at this dataset scale.

### 4.2 Feature importance (Top 5)
| Rank | Feature | Importance | Business signal |
|------|---------|-----------|-----------------|
| 1 | `last_year_renewed` | 0.290 | Past renewal predicts future renewal |
| 2 | `missed_payments` | 0.141 | Disengagement leading indicator |
| 3 | `annual_premium` | 0.098 | Higher premium → stickier customer |
| 4 | `premium_change_pct` | 0.080 | Large price rises trigger churn |
| 5 | `previous_renewals_count` | 0.054 | Cumulative loyalty |

**Four of the top five features are behavioural**, not demographic. The model is learning genuine mechanisms, not spurious patterns.

### 4.3 Iteration History
- **First iteration:** Static attributes only. ROC-AUC **0.70**. Diagnosis: missing behavioural signals; `annual_premium` dominated as a segment proxy.
- **Second iteration:** Added four behavioural features. Corrected generator to retain lapsed customers. ROC-AUC improved from **0.70 → 0.81** (+0.11).
- **Lesson:** Feature engineering had more impact than model architecture.

---

## 5. Business Impact & Financial Translation

While ROC-AUC confirms the model's statistical validity, the business requires a financial translation of its predictive power. 

**Calculation Methodology:**
1. **Portfolio CLV:** Based on the parallel CLV engagement, the 12-month projected portfolio value of this 5,000-customer book is **£3.33M**.
2. **Model Targeting:** The classifier allows the retention team to target the top 20% highest-risk customers with high-touch interventions (e.g., payment plans, loyalty offers).
3. **Expected Lift:** Industry-standard retention campaigns yield a conservative **5% lift** in saved customers within the targeted cohort.
4. **ROI Estimate:** £3.33M × 5% = **~£166,000 in protected annual revenue**.

This figure serves as the baseline business case for deploying the model. 

---

## 6. Limitations

**1. Synthetic data** — reproduces realistic renewal behaviour but is not real client data. Real-world distributions may differ.

**2. Financial assumptions** — The £166,000 impact estimate is directional. It assumes a 5% campaign success rate. Actual ROI requires A/B testing in production (e.g., a control group that receives no outreach vs. a treatment group driven by model scores).

**3. Random split, not temporal** — The train/test split is random. A production model should be validated on a holdout period (e.g., train on 2018–2023, test on 2024–2025).

**4. No hyperparameter tuning** — Sensible defaults were used. Production deployment would tune settings via cross-validation.

**5. Threshold assumption** — The reported threshold maximises F1. In production, the threshold should align with the business cost of false positives (wasted retention offer) vs false negatives (lost customer).

---

## 7. Reproducibility

The full pipeline runs end-to-end from a clean clone:

```bash
git clone https://github.com/YinkaAdebimpe/guardianshield-renewal.git
cd guardianshield-renewal
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with Postgres credentials
cd src
python generate_renewal_data.py
python load_to_postgres.py
cd ..
jupyter lab
```

**Environment requirements:** Python 3.12+, PostgreSQL 14+, database `guardianshield_db`.

---
*Prepared as part of the OkYeahInsight Analytics portfolio — a three-engagement consulting series for GuardianShield Insurance.*
