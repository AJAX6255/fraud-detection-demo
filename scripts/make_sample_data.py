"""Generate a small Sparkov-schema CSV for smoke tests and offline development.

Fraud rows are injected with classic patterns: night hours, unusually high
amounts, and merchants far from the customer's home — so the model has real
signal to learn even without downloading the Kaggle dataset.

Usage: python -m scripts.make_sample_data --n 20000 --out data/sample_transactions.csv
"""
import argparse
import os
import uuid

import numpy as np
import pandas as pd

FIRST = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
         "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
         "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen"]
LAST = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez"]
JOBS = ["Engineer", "Teacher", "Nurse", "Accountant", "Designer", "Manager",
        "Analyst", "Developer", "Chef", "Electrician"]
CATS = ["gas_transport", "grocery_pos", "home", "shopping_pos", "kids_pets",
        "shopping_net", "entertainment", "food_dining", "personal_care",
        "health_fitness", "misc_pos", "misc_net", "travel"]
CAT_W = [.12, .15, .09, .10, .05, .06, .07, .10, .05, .05, .07, .04, .05]
CITIES = [("Springfield", "IL", 39.78, -89.65, 116000),
          ("Austin", "TX", 30.27, -97.74, 965000),
          ("Denver", "CO", 39.74, -104.99, 715000),
          ("Portland", "OR", 45.52, -122.68, 650000),
          ("Miami", "FL", 25.76, -80.19, 440000),
          ("Columbus", "OH", 39.96, -82.99, 905000)]


def generate(n=20000, out="data/sample_transactions.csv", days=180, seed=42):
    rng = np.random.default_rng(seed)
    customers = []
    for _ in range(300):
        cname, stt, la, lo, pop = CITIES[rng.integers(len(CITIES))]
        customers.append(dict(
            cc_num=int(rng.integers(4_000_000_000_000_000, 4_600_000_000_000_000)),
            first=str(rng.choice(FIRST)), last=str(rng.choice(LAST)),
            gender=str(rng.choice(["M", "F"])), city=cname, state=stt,
            zip=str(rng.integers(10000, 99999)),
            lat=la + rng.normal(0, 0.05), long=lo + rng.normal(0, 0.05),
            city_pop=pop, street=f"{rng.integers(1, 9999)} {rng.choice(LAST)} St",
            job=str(rng.choice(JOBS)),
            dob=str(pd.Timestamp("1955-01-01") + pd.Timedelta(days=int(rng.integers(0, 16000))))[:10]))

    rows, start = [], pd.Timestamp("2020-01-01")
    for _ in range(n):
        c = customers[rng.integers(len(customers))]
        ts = start + pd.Timedelta(minutes=int(rng.integers(0, days * 24 * 60)))
        is_fraud = int(rng.random() < 0.008)
        if is_fraud:  # night-time, high amount, far-away merchant
            ts = ts.replace(hour=int(rng.choice([0, 1, 2, 3, 23])))
            amt = float(np.exp(rng.normal(6.5, 0.6)))
            mlat = c["lat"] + rng.uniform(2, 8) * rng.choice([-1, 1])
            mlong = c["long"] + rng.uniform(2, 8) * rng.choice([-1, 1])
        else:
            amt = float(np.exp(rng.normal(3.4, 1.0)))
            mlat = c["lat"] + rng.normal(0, 0.1)
            mlong = c["long"] + rng.normal(0, 0.1)
        rows.append(dict(
            trans_date_trans_time=str(ts), cc_num=c["cc_num"],
            merchant=f"fraud_{rng.choice(LAST)} Inc" if is_fraud else f"{rng.choice(LAST)} Store",
            category=str(rng.choice(CATS, p=CAT_W)), amt=round(amt, 2),
            first=c["first"], last=c["last"], gender=c["gender"],
            street=c["street"], city=c["city"], state=c["state"], zip=c["zip"],
            lat=round(c["lat"], 6), long=round(c["long"], 6), city_pop=c["city_pop"],
            job=c["job"], dob=c["dob"], trans_num=uuid.uuid4().hex,
            unix_time=int(ts.timestamp()),
            merch_lat=round(float(mlat), 6), merch_long=round(float(mlong), 6),
            is_fraud=is_fraud))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default="data/sample_transactions.csv")
    a = ap.parse_args()
    path = generate(a.n, a.out)
    print(f"wrote {a.n} rows -> {path}")


if __name__ == "__main__":
    main()
