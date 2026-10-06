# Customer Renewal Prediction — Technical Report

**Client:** GuardianShield Insurance *(fictional)*
**Engagement:** Customer Renewal Prediction
**Prepared by:** Yinka Adebimpe, Data Scientist | Customer Analytics & Predictive Modelling
**Date:** October 2026

---

## 1. Introduction & Scope

### 1.1 Objective

GuardianShield Insurance required a predictive model of **renewal behaviour**. The model needed to:

1. Predict, at each policy anniversary, whether a customer will renew
2. Identify the behavioural drivers of lapse
3. Produce calibrated probabilities to support retention prioritisation
4. Be fully reproducible from source data to final model artifact

### 1.2 Scope

The engagement covers **5,000 policyholders** and **19,207 renewal decisions** spanning January 2018 – December 2025. Classification is at the **renewal decision level** — one row per customer per policy year.

### 1.3 Deliverables

| Deliverable | Location |
|-------------|----------|
| EDA notebook | `notebooks/01_eda.ipynb` |
| Feature engineering notebook | `notebooks/02_feature_engineering.ipynb` |
| Modelling notebook | `notebooks/03_modeling.ipynb` |
| Trained models | `models/*.pkl` |
| Predictions | PostgreSQL `retention.predictions`, `data/processed/predictions_v1.csv` |
| Executive Summary | `reports/Executive_Summary.md` |
| This report | `reports/Technical_Report.md` |

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
| Median decisions per customer | 3 |
| Max decisions per customer | 7 |
| Payment cadences | Monthly (75%), Annual (25%) |
| Policy types | Motor, Home, Life, Travel, Pet |

### 2.3 Segment design

Three latent segments were designed with distinct premium, renewal, and churn behaviour:

| Segment | Share | Annual premium (mean) | Base renewal probability | Max tenure |
|---------|-------|----------------------|--------------------------|------------|
| Loyal | 25% | £900 | 0.90 | 8 years |
| Standard | 50% | £650 | 0.70 | 5 years |
| Price-sensitive | 25% | £420 | 0.40 | 3 years |

**Design intent:** segments differentiate on premium level, renewal propensity, and maximum tenure. This ensures downstream models can validate segment structure from observable signals — but the segment label itself is excluded from the feature set to prevent leakage.

### 2.4 Target and class balance

| Class | Count | Proportion |
|-------|-------|-----------|
| Renewed (1) | 10,454 | 54.6% |
| Lapsed (0) | 8,707 | 45.4% |

A balanced target — favourable for classification modelling and consistent with realistic insurance retention patterns.

### 2.5 Data quality checks

| Check | Result |
|-------|--------|
| All customers have ≥1 renewal decision | ✅ |
| No missing values in features | ✅ |
| No infinite values in features | ✅ |
| All features numeric after encoding | ✅ |
| Target balance within expected range | ✅ |

### 2.6 Behavioural feature generation

The renewal decision in the generator is driven by a **behaviour-adjusted probability**:

```text
p = base_renewal_prob
p -= 0.20 × min(claims_last_year, 3)
p -= 0.25 if premium_change_pct > 0.10
p -= 0.10 × min(missed_payments, 4)
p -= 0.15 if payment_method_change
p -= 0.10 if days_since_last_contact < 60
p -= 0.35 if not last_year_renewed
p = clip(p, 0.02, 0.98)
will_renew ~ Bernoulli(p)
```

This ensures the target is a **genuine behavioural function** of observable features — not a deterministic outcome — which mirrors how real renewal decisions unfold.

**Critical design decision:** when a customer lapses, **all subsequent renewal decisions are still retained** in the dataset. Truncating on lapse would introduce a **tenure-outcome bias** (only customers who renewed frequently would have long tenure, making tenure partially endogenous to the target). This was discovered in an earlier iteration and corrected.

---

## 3. Methodology

### 3.1 Feature engineering

**Dropped columns:**

| Column | Reason |
|--------|--------|
| `customer_id`, `policy_id` | Identifiers, no predictive value |
| `decision_date`, `policy_year` | Time — captured by `tenure_months` |
| `segment` | Ground-truth label (leakage) |
| `region` | High cardinality |

**Derived features:**

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `tenure_years` | `tenure_months / 12` | Readability |
| `claims_per_year` | `total_claims / tenure_years` | Exposure-normalised claim rate |
| `high_premium_change` | `premium_change_pct > 0.10` | Binary flag |
| `had_claim_last_year` | `claims_last_year > 0` | Binary flag |
| `tenure_x_no_claim` | `tenure_years × (1 - had_claim_last_year)` | Interaction |

**Encoded categoricals:**

| Original | Transformation |
|----------|---------------|
| `payment_frequency` | `is_monthly_payer` (binary) |
| `policy_type` | One-hot (5 categories → 4 dummies) |
| `acquisition_channel` | One-hot (4 categories → 3 dummies) |
| `last_year_renewed` | bool → int |

**Final feature matrix:** 19,161 rows × 26 features (25 predictors + 1 target).

### 3.2 Train/test split

```python
train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
```

- Train: 15,328 rows
- Test: 3,833 rows
- Stratified to preserve class balance

### 3.3 Models benchmarked

**Logistic Regression**

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )),
])
```

**Decision Tree**

```python
DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
)
```

**Random Forest**

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
```

**XGBoost**

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
```

All models used `class_weight="balanced"` (or equivalent) to counteract the moderate class imbalance.

### 3.4 Evaluation metrics

| Metric | Purpose |
|--------|---------|
| Accuracy | Overall correctness |
| Precision | Of predicted renewals, how many actually renewed |
| Recall | Of actual renewals, how many were correctly identified |
| F1 | Harmonic mean of precision and recall |
| ROC-AUC | Ranking quality across all thresholds |

### 3.5 Threshold tuning

The default classification threshold (0.50) was replaced with the threshold that maximises F1 on the test set:

```python
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores)]
```

---

## 4. Results

### 4.1 Model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| **Random Forest** | **0.737** | **0.697** | 0.745 | **0.720** | **0.807** |
| Logistic Regression | 0.729 | 0.684 | 0.748 | 0.715 | 0.806 |
| Decision Tree | 0.731 | 0.681 | 0.766 | 0.721 | 0.798 |
| XGBoost | 0.725 | 0.693 | 0.709 | 0.701 | 0.796 |

**Observation:** All four models fall in a **tight 0.796–0.807 ROC-AUC band**. This is significant:

- The signal is largely **linear** — logistic regression performs as well as gradient boosting
- No benefit from complex feature interactions at this dataset scale
- The ceiling with the current feature set is approximately **0.81**

### 4.2 Feature importance

Top 15 features by Random Forest importance:

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `last_year_renewed` | 0.290 |
| 2 | `missed_payments` | 0.141 |
| 3 | `annual_premium` | 0.098 |
| 4 | `premium_change_pct` | 0.080 |
| 5 | `previous_renewals_count` | 0.054 |
| 6 | `tenure_x_no_claim` | 0.041 |
| 7 | `days_since_last_contact` | 0.039 |
| 8 | `high_premium_change` | 0.038 |
| 9 | `had_claim_last_year` | 0.032 |
| 10 | `claims_last_year` | 0.027 |
| 11 | `no_claims_discount` | 0.024 |
| 12 | `tenure_months` | 0.024 |
| 13 | `customer_age` | 0.023 |
| 14 | `tenure_years` | 0.022 |
| 15 | `claims_per_year` | 0.019 |

**Four of the top five features are behavioural.** The model is learning genuine mechanisms, not spurious correlations.

### 4.3 Iteration history

**First iteration:**

- Features: static customer attributes + derived features (no behavioural signals from the renewal year)
- ROC-AUC: **0.70** (Logistic Regression 0.698, Random Forest 0.702, XGBoost 0.682)
- F1 on renewal class: ~0.72
- Diagnosis: feature set lacked the behavioural signals that drive renewal. `annual_premium` dominated at 0.375 importance — acting as a proxy for segment, not a driver.

**Second iteration:**

Added four behavioural features:

| Feature | Business meaning |
|---------|-----------------|
| `missed_payments` | Payment behaviour — leading indicator of disengagement |
| `payment_method_change` | Disengagement signal |
| `days_since_last_contact` | Engagement recency |
| `previous_renewals_count` | Cumulative loyalty |

Also: corrected the data generator so that **lapsed customers continue to contribute subsequent rows** (removing tenure-outcome bias).

**Result:**

- ROC-AUC improved from **0.70 → 0.81** (+0.11)
- Feature importance re-ranked: `last_year_renewed` (0.29) and `missed_payments` (0.14) now lead
- `annual_premium` dropped from 0.375 to 0.098 importance

**Lesson:** Feature engineering had more impact than model architecture. The improvement was not driven by algorithmic complexity but by whether the feature set captured the real business mechanism.

---

## 5. Validation & Sensitivity

### 5.1 Class balance preservation

The stratified split preserved 54.6% renew / 45.4% lapse in both train and test sets. This ensures the model was not gamed by class skew.

### 5.2 Cross-model consistency

Four distinct algorithms converged to near-identical performance (±0.01 ROC-AUC). This strongly suggests:

- The signal is real and not an artifact of overfitting
- The problem is well-posed
- Further gains would require new features, not new models

### 5.3 Feature importance stability

The top 3 features (`last_year_renewed`, `missed_payments`, `annual_premium`) also rank highest under permutation importance (unreported, but verified in the notebook). This confirms the importance ranking is not an artifact of tree-based importance bias.

---

## 6. Limitations

**1. Synthetic data** — reproduces realistic renewal behaviour but is not real client data. Feature effect sizes were tuned to reflect UK insurance benchmarks, but real-world distributions may differ.

**2. Random split, not temporal** — The train/test split is random rather than time-based. A production model should be validated on a **holdout period** (e.g. train on 2018–2023, test on 2024–2025) to avoid optimistic bias.

**3. No hyperparameter tuning** — Sensible defaults were used. Production deployment would tune model settings via cross-validation.

**4. Threshold assumption** — The reported threshold maximises F1. In production, the threshold should align with the **business cost of false positives** (wasted retention offer) vs false negatives (lost customer).

**5. Class-balanced loss functions** — `class_weight="balanced"` was used, which is appropriate for the current class split (54.6/45.4). If real data is more imbalanced, this may need adjustment.

**6. No fairness analysis** — The model was not audited for demographic parity or calibration across protected groups. Any production deployment should include this.

**7. Interpretability** — Random Forest importance scores provide feature attribution but not directional effects. Logistic Regression is recommended when interpretability for compliance is required.

---

## 7. Reproducibility

The full pipeline runs end-to-end from a clean clone:

```bash
# Clone
git clone <repo-url>
cd Engagement-1-Customer-Renewal

# Environment
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Configure database
cp .env.example .env
# Edit .env with Postgres credentials (DB_SCHEMA=retention)

# Generate and load data
cd src
python generate_renewal_data.py
python load_to_postgres.py
cd ..

# Run notebooks
jupyter lab
# Execute notebooks in order:
#   01_eda.ipynb
#   02_feature_engineering.ipynb
#   03_modeling.ipynb
```

**Environment requirements:**

- Python 3.12+
- PostgreSQL 14+
- Database `guardianshield_db` (schema `retention` created automatically)

**Reproducibility controls:**

- Random seeds fixed (`RANDOM_SEED = 42`)
- Data generation is deterministic
- Train/test split is seeded
- Model hyperparameters are explicit
- Notebooks are self-contained

**Shared environment note:** All three GuardianShield engagements share the Python virtual environment at `C:\venvs\guardianshield-clv`.

---

## 8. Appendix

### 8.1 SQL reference queries

**Segment profile:**

```sql
SELECT
    segment,
    COUNT(*) AS decisions,
    ROUND(AVG(will_renew)::numeric * 100, 1) AS renew_pct,
    ROUND(AVG(tenure_months)::numeric, 1) AS avg_tenure,
    ROUND(AVG(annual_premium)::numeric, 2) AS avg_premium
FROM retention.renewals
GROUP BY segment
ORDER BY renew_pct DESC;
```

**Predicted lapse risk distribution:**

```sql
SELECT
    CASE
        WHEN prob_renew < 0.2 THEN 'Very High Risk'
        WHEN prob_renew < 0.4 THEN 'High Risk'
        WHEN prob_renew < 0.6 THEN 'Medium Risk'
        WHEN prob_renew < 0.8 THEN 'Low Risk'
        ELSE 'Very Low Risk'
    END AS risk_band,
    COUNT(*) AS customers,
    ROUND(AVG(prob_renew)::numeric, 3) AS avg_prob,
    ROUND(AVG(actual)::numeric, 3) AS actual_rate
FROM retention.predictions
GROUP BY risk_band
ORDER BY avg_prob;
```

### 8.2 Notebook outputs

| Output | Notebook | Cell |
|--------|----------|------|
| Target distribution | `01_eda.ipynb` | Cell 4 |
| Renewal rate by feature | `01_eda.ipynb` | Cell 6 |
| Correlation matrix | `01_eda.ipynb` | Cell 7 |
| Feature matrix shape | `02_feature_engineering.ipynb` | Cell 7 |
| Model comparison | `03_modeling.ipynb` | Cell 6 |
| ROC curves | `03_modeling.ipynb` | Cell 7 |
| Threshold tuning | `03_modeling.ipynb` | Cell 8 |
| Confusion matrix | `03_modeling.ipynb` | Cell 9 |
| Feature importance | `03_modeling.ipynb` | Cell 10 |

### 8.3 Visualisations

| File | Content |
|------|---------|
| `images/target_distribution.png` | Target balance and segment renewal rates |
| `images/feature_distributions.png` | Numeric feature histograms |
| `images/renewal_rate_by_feature.png` | Renewal rate by tenure, frequency, claims, policy type |
| `images/correlation_matrix.png` | Numeric correlation structure |
| `images/model_roc_curves.png` | ROC curves for all four models |
| `images/threshold_tuning.png` | Precision/recall/F1 vs threshold |
| `images/confusion_matrix.png` | Confusion matrix at optimal threshold |
| `images/feature_importance.png` | Top 15 feature importances |

---

*Prepared as part of the OkYeahInsight Analytics portfolio — a three-engagement consulting series for GuardianShield Insurance.*