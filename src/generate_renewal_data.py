# ============================================
# GUARDIANSHIELD INSURANCE
# ENGAGEMENT 1 — RENEWAL DATA GENERATION
# ============================================
# Produces one row per renewal decision per customer.
# Target: will_renew (0/1).
# ============================================

from datetime import date, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from config import RAW_DATA_DIR, RANDOM_SEED, N_CUSTOMERS

# ============================================
# SETUP
# ============================================

np.random.seed(RANDOM_SEED)
fake = Faker("en_GB")
Faker.seed(RANDOM_SEED)

SIMULATION_START = date(2018, 1, 1)
SIMULATION_END = date(2025, 12, 31)

POLICY_TYPES = ["Motor", "Home", "Life", "Travel", "Pet"]
POLICY_TYPE_WEIGHTS = [0.40, 0.25, 0.10, 0.15, 0.10]

ACQUISITION_CHANNELS = ["Online", "Broker", "Direct", "Comparison Site"]
ACQUISITION_WEIGHTS = [0.35, 0.20, 0.15, 0.30]

# Segment design (matches Engagement 2)
SEGMENTS = {
    "loyal": {
        "prob": 0.25,
        "premium_mean": 900,
        "renewal_prob": 0.90,
        "max_years": 8,
    },
    "standard": {
        "prob": 0.50,
        "premium_mean": 650,
        "renewal_prob": 0.70,
        "max_years": 5,
    },
    "price_sensitive": {
        "prob": 0.25,
        "premium_mean": 420,
        "renewal_prob": 0.40,
        "max_years": 3,
    },
}


# ============================================
# HELPERS
# ============================================

def add_months(d: date, months: int) -> date:
    month = d.month - 1 + months
    year = d.year + month // 12
    month = month % 12 + 1
    day = min(d.day, 28)
    return date(year, month, day)


def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=int(np.random.randint(0, delta + 1)))


def assign_segment() -> str:
    segments = list(SEGMENTS.keys())
    probs = [SEGMENTS[s]["prob"] for s in segments]
    return np.random.choice(segments, p=probs)


# ============================================
# CUSTOMER GENERATION
# ============================================

def generate_customer(customer_index: int) -> dict:
    customer_id = f"GS-{customer_index:06d}"
    segment = assign_segment()
    seg = SEGMENTS[segment]

    earliest_start = SIMULATION_START
    policy_start = random_date(earliest_start, SIMULATION_END - timedelta(days=365 * 2))

    payment_frequency = np.random.choice(["Monthly", "Annual"], p=[0.75, 0.25])

    annual_premium = float(
        np.random.lognormal(mean=np.log(seg["premium_mean"]), sigma=0.35)
    )

    return {
        "customer_id": customer_id,
        "policy_id": f"POL-{customer_index:06d}-01",
        "policy_start_date": policy_start,
        "policy_type": np.random.choice(POLICY_TYPES, p=POLICY_TYPE_WEIGHTS),
        "payment_frequency": payment_frequency,
        "annual_premium": round(annual_premium, 2),
        "region": fake.county(),
        "postcode": fake.postcode(),
        "customer_age": int(np.clip(np.random.normal(45, 14), 18, 80)),
        "acquisition_channel": np.random.choice(
            ACQUISITION_CHANNELS, p=ACQUISITION_WEIGHTS
        ),
        "excess_amount": float(np.random.choice([0, 100, 250, 500, 750])),
        "no_claims_discount": round(float(np.random.uniform(0, 0.7)), 2),
        "segment": segment,
        "renewal_prob": seg["renewal_prob"],
        "max_years": seg["max_years"],
    }

    

# ============================================
# RENEWAL DECISION SIMULATION
# ============================================

def simulate_renewal_history(customer: dict) -> tuple[list, bool]:
    """
    Simulate renewal decisions at each policy anniversary.
    Returns: (list of renewal records, eventually_churned flag)
    """
    records = []
    policy_start = customer["policy_start_date"]
    renewal_prob = customer["renewal_prob"]
    max_years = customer["max_years"]

    # Baseline annual premium and behavioural drift
    baseline_premium = customer["annual_premium"]
    prior_year_renewed = True
    total_claims = 0
    eventually_churned = False

    for year_idx in range(1, max_years + 1):
        anniversary = add_months(policy_start, year_idx * 12)
        if anniversary > SIMULATION_END:
            break

        tenure_months = year_idx * 12

        # Premium drift: small increase each year; bigger if claims
        year_claims = int(np.random.poisson(0.15))
        total_claims += year_claims

        premium_change_pct = float(np.random.normal(0.05, 0.03))  # ~5% ± 3%
        if year_claims > 0:
            premium_change_pct += 0.08 * year_claims
        annual_premium = round(baseline_premium * (1 + premium_change_pct), 2)

        # Dynamic renewal probability (behaviour-adjusted)
        p = renewal_prob
        if year_claims > 0:
            p -= 0.05 * min(year_claims, 3)   # claims → higher churn
        if premium_change_pct > 0.10:
            p -= 0.10                          # big price increase → churn
        if not prior_year_renewed:
            p -= 0.20                          # past lapsed → less sticky
        p = float(np.clip(p, 0.05, 0.98))

        # Renewal outcome
        will_renew = bool(np.random.random() < p)

        records.append({
            "customer_id": customer["customer_id"],
            "policy_id": customer["policy_id"],
            "decision_date": anniversary,
            "policy_year": year_idx,
            "tenure_months": tenure_months,
            "annual_premium": annual_premium,
            "premium_change_pct": round(premium_change_pct, 3),
            "claims_last_year": year_claims,
            "total_claims": total_claims,
            "payment_frequency": customer["payment_frequency"],
            "policy_type": customer["policy_type"],
            "acquisition_channel": customer["acquisition_channel"],
            "customer_age": customer["customer_age"],
            "region": customer["region"],
            "excess_amount": customer["excess_amount"],
            "no_claims_discount": customer["no_claims_discount"],
            "last_year_renewed": prior_year_renewed,
            "segment": customer["segment"],
            "will_renew": int(will_renew),
        })

        if not will_renew:
            eventually_churned = True
            break   # customer lapses; no further renewals

        prior_year_renewed = True

    return records, eventually_churned


# ============================================
# BUILD DATASET
# ============================================

def build_dataset(n_customers: int = N_CUSTOMERS):
    print(f"Generating renewal history for {n_customers} customers...")

    customer_records = []
    renewal_records = []

    for i in range(1, n_customers + 1):
        cust = generate_customer(i)
        renewals, churned = simulate_renewal_history(cust)

        if not renewals:
            continue

        customer_records.append({
            "customer_id": cust["customer_id"],
            "policy_id": cust["policy_id"],
            "policy_type": cust["policy_type"],
            "payment_frequency": cust["payment_frequency"],
            "region": cust["region"],
            "postcode": cust["postcode"],
            "customer_age": cust["customer_age"],
            "acquisition_channel": cust["acquisition_channel"],
            "excess_amount": cust["excess_amount"],
            "no_claims_discount": cust["no_claims_discount"],
            "segment": cust["segment"],
            "policy_start_date": cust["policy_start_date"],
            "eventually_churned": churned,
        })

        renewal_records.extend(renewals)

    customers_df = pd.DataFrame(customer_records)
    renewals_df = pd.DataFrame(renewal_records)

    print(f"Customers:          {len(customers_df):,}")
    print(f"Renewal decisions:  {len(renewals_df):,}")
    print(f"\nTarget distribution (will_renew):")
    print(renewals_df["will_renew"].value_counts(normalize=True).round(3))
    print(f"\nRenewal rate by segment:")
    print(
        renewals_df.groupby("segment")["will_renew"]
        .agg(["mean", "count"])
        .round(3)
    )
    print(f"\nRenewal decisions per customer:")
    print(renewals_df.groupby("customer_id").size().describe().round(2))

    return customers_df, renewals_df


if __name__ == "__main__":
    customers_df, renewals_df = build_dataset()

    customers_path = RAW_DATA_DIR / "customers.csv"
    renewals_path = RAW_DATA_DIR / "renewals.csv"

    customers_df.to_csv(customers_path, index=False)
    renewals_df.to_csv(renewals_path, index=False)

    print(f"\nSaved: {customers_path}")
    print(f"Saved: {renewals_path}")