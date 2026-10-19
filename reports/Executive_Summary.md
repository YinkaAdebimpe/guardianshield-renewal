# Customer Renewal Prediction — Executive Summary

**Client:** GuardianShield Insurance *(fictional)*  
**Engagement:** Customer Renewal Prediction  
**Prepared by:** Yinka Adebimpe, Data Scientist | Customer Analytics & Predictive Modelling  
**Date:** October 2026

---

## 1. Overview

GuardianShield Insurance renews **5,000 policyholders** each year. Leadership wanted to answer a single question: **which customers are most likely to lapse at renewal, and what drives that behaviour?**

The result is a **binary classification model** that predicts, at each policy anniversary, whether a customer will renew. The model achieves **ROC-AUC 0.81** and identifies four behavioural triggers that a retention team can act on before the renewal date.

---

## 2. Key Findings

**1. Renewal is driven by behaviour, not demographics**
The four strongest predictors are all behavioural: last year's renewal decision, missed payments, premium change size, and cumulative renewals. Age, region, excess, and policy type add almost nothing to prediction.

**2. Missed payments are the strongest negative signal**
Customers with missed payments in the previous year are significantly more likely to lapse. This is a **leading indicator of disengagement** — and it is **controllable** through outreach.

**3. Large premium increases trigger churn**
Customers facing premium increases above 10% are markedly more likely to leave. This is the **largest controllable churn driver** on the pricing side.

**4. Past renewal predicts future renewal**
Customers who renewed last year are far more likely to renew again. This is the single most important feature in the model, with importance **0.29** (next-highest is 0.14).

**5. Model performance reaches 0.81 ROC-AUC**
Four classifiers were benchmarked. All four models cluster in a tight band, indicating the signal is largely linear and requires no exotic modelling.

---

## 3. Segment Profile

Underlying customer behaviour clusters into three distinct renewal profiles:

| Segment | Decisions | Renewal rate | Avg tenure | Avg premium |
|---------|-----------|-------------|------------|-------------|
| Loyal | 5,828 | 84.8% | 32.9 months | £1,019 |
| Standard | 9,741 | 61.8% | 24.7 months | £735 |
| Price-sensitive | 3,638 | 28.8% | 17.0 months | £475 |

---

## 4. Recommendations & Business Impact

### Priority 1 — Trigger outreach on missed payments
**Action:** Any customer with one or more missed payments in the last year should be flagged for retention contact before their next renewal date.
**Impact:** Missed payments are the strongest controllable churn signal in the model (importance 0.14). A proactive call and payment-plan offer could recover a meaningful share of these customers.

### Priority 2 — Pre-empt large premium increases
**Action:** Customers facing premium increases above 10% should receive a retention offer (loyalty discount, added cover, or renewal incentive) before the renewal date.
**Impact:** Premium increase magnitude is the fourth-most-important feature. This is the largest **pricing-side** churn lever, entirely under GuardianShield's control.

### Priority 3 — Recognise long-tenure customers
**Action:** Customers with 3+ prior renewals should be flagged for loyalty recognition — not necessarily discount, but VIP treatment.
**Impact:** Cumulative renewals is the fifth-strongest feature. These customers are highly likely to renew, so retention spend is more about protecting the relationship than preventing churn.

### Business Impact Estimate
By targeting the high-risk segments identified by this model with retention campaigns, GuardianShield protects its revenue base. **At a projected portfolio CLV of £3.33M, a conservative 5% retention lift across targeted segments translates to approximately £166,000 in protected annual revenue.**

---

## 5. Method Summary

Three notebook stages:
1. **EDA** — established target balance (54.6% renewal / 45.4% lapse) and confirmed three strong signals: tenure, claims, and premium change.
2. **Feature engineering** — encoded categoricals, created derived features, and added four behavioural features.
3. **Modelling** — benchmarked four classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost).

### Iteration story
The initial model, trained without the four added behavioural features, achieved **ROC-AUC 0.70**. Diagnosing the shortfall led to adding four behavioural signals, which raised ROC-AUC to **0.81** — an improvement of roughly **+0.10 across every model**.
**Lesson:** Feature engineering had more impact than model selection. Once the right behavioural signals were available, all four classifiers converged to nearly identical performance.

---

## 6. Scope & Limitations

**Synthetic data** — This engagement uses fully reproducible synthetic data. Feature effect sizes were tuned to reflect UK insurance benchmarks, but the model should be re-validated on real client data.

**Business Impact Assumptions** — Financial impact estimates (such as the £166,000 retention lift) are based on a conservative 5% targeted campaign success rate applied to the model's baseline predictions. They represent a directional business case rather than a guaranteed outcome and require A/B testing in production to validate.

**Random split** — The train/test split is random rather than time-based. A production model should be validated on a holdout period.

**Threshold assumption** — The optimal threshold maximises F1. Production deployment should choose a threshold aligned with the business cost of false positives (wasted retention offers) vs false negatives (lost customers).

---

## 7. Appendix

**Data**
- 5,000 customers, 19,207 renewal decisions
- Window: January 2018 – December 2025
- Target balance: 54.6% renew / 45.4% lapse

**Model performance (Random Forest)**
- ROC-AUC: **0.807** | Precision: 0.697 | Recall: 0.745 | F1: 0.720

**Repository**
- Full code, notebooks, and documentation: [github.com/YinkaAdebimpe/guardianshield-renewal](https://github.com/YinkaAdebimpe/guardianshield-renewal)

---
*Prepared as part of the OkYeahInsight Analytics portfolio — a three-engagement consulting series for GuardianShield Insurance.*
