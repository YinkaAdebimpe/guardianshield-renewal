# ============================================
# GUARDIANSHIELD INSURANCE
# ENGAGEMENT 1 — LOAD DATA TO POSTGRES
# ============================================

import pandas as pd
from sqlalchemy import create_engine, text

from config import DB_URL, DB_SCHEMA, RAW_DATA_DIR


def load_csvs():
    customers = pd.read_csv(RAW_DATA_DIR / "customers.csv")
    renewals = pd.read_csv(RAW_DATA_DIR / "renewals.csv")

    customers["policy_start_date"] = pd.to_datetime(
        customers["policy_start_date"]
    ).dt.date
    renewals["decision_date"] = pd.to_datetime(
        renewals["decision_date"]
    ).dt.date

    return customers, renewals


def ensure_schema(engine):
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {DB_SCHEMA};"))
        print(f"Schema '{DB_SCHEMA}' ready.")


def write_table(df: pd.DataFrame, table_name: str, engine):
    df.to_sql(
        name=table_name,
        con=engine,
        schema=DB_SCHEMA,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=1000,
    )
    print(f"Wrote {len(df):,} rows → {DB_SCHEMA}.{table_name}")


if __name__ == "__main__":
    print("Reading CSVs...")
    customers, renewals = load_csvs()

    print(f"Loaded customers: {len(customers):,}")
    print(f"Loaded renewals:  {len(renewals):,}")

    print("\nConnecting to Postgres...")
    engine = create_engine(DB_URL)

    ensure_schema(engine)

    print("\nWriting tables...")
    write_table(customers, "customers", engine)
    write_table(renewals, "renewals", engine)

    with engine.connect() as conn:
        cust_count = conn.execute(
            text(f"SELECT COUNT(*) FROM {DB_SCHEMA}.customers;")
        ).scalar()
        ren_count = conn.execute(
            text(f"SELECT COUNT(*) FROM {DB_SCHEMA}.renewals;")
        ).scalar()

    print(f"\nVerification:")
    print(f"  {DB_SCHEMA}.customers:  {cust_count:,} rows")
    print(f"  {DB_SCHEMA}.renewals:   {ren_count:,} rows")

    print("\n✓ Data loaded successfully.")