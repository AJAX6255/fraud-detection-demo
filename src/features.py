"""Feature engineering for the Sparkov credit-card fraud dataset.

Schema expected (columns): trans_date_trans_time, cc_num, merchant, category,
amt, first, last, gender, street, city, state, zip, lat, long, city_pop, job,
dob, trans_num, unix_time, merch_lat, merch_long, is_fraud
"""
import numpy as np
import pandas as pd

BASE_FEATURES = [
    "amt", "amt_log", "hour", "day_of_week", "is_weekend", "is_night",
    "distance_km", "city_pop_log", "age", "gender_m", "amt_zscore",
    "tx_count_24h", "lat", "long", "merch_lat", "merch_long",
]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between customer home and merchant."""
    r = 6371.0
    lat1, lon1 = lat1.astype(float), lon1.astype(float)
    lat2, lon2 = lat2.astype(float), lon2.astype(float)
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _velocity_24h(df):
    """Count of transactions per card in the trailing 24h window."""
    return (
        df.sort_values("trans_date_trans_time")
        .set_index("trans_date_trans_time")
        .groupby("cc_num")["amt"]
        .rolling("24h")
        .count()
        .rename("tx_count_24h")
        .reset_index()
        .drop_duplicates(["cc_num", "trans_date_trans_time"])
    )


def engineer(df, amt_stats=None, fit=False):
    """Return (feature_df, amt_stats).

    fit=True  -> compute per-customer amount stats from this data (training set).
    fit=False -> reuse the training stats passed via `amt_stats` (scoring time).
    """
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    ts = df["trans_date_trans_time"]

    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 4)).astype(int)

    df["amt"] = df["amt"].astype(float)
    df["amt_log"] = np.log1p(df["amt"])
    df["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])
    df["city_pop_log"] = np.log1p(pd.to_numeric(df["city_pop"], errors="coerce").fillna(0).clip(lower=0))

    dob = (pd.to_datetime(df["dob"], errors="coerce") if "dob" in df.columns
           else pd.Series(pd.NaT, index=df.index))
    df["age"] = ((ts - dob).dt.days / 365.25).fillna(40.0)
    df["gender_m"] = (df["gender"].astype(str).str.upper() == "M").astype(int)

    if fit:
        g = df.groupby("cc_num")["amt"]
        amt_stats = pd.DataFrame({"cust_amt_mean": g.mean(),
                                  "cust_amt_std": g.std().fillna(1.0)})
    df = df.merge(amt_stats, left_on="cc_num", right_index=True, how="left")
    df["cust_amt_mean"] = df["cust_amt_mean"].fillna(df["amt"].median())
    df["cust_amt_std"] = df["cust_amt_std"].fillna(1.0).replace(0, 1.0)
    df["amt_zscore"] = (df["amt"] - df["cust_amt_mean"]) / df["cust_amt_std"]

    df = df.merge(_velocity_24h(df), on=["cc_num", "trans_date_trans_time"], how="left")
    df["tx_count_24h"] = df["tx_count_24h"].fillna(1)

    cats = pd.get_dummies(df["category"].astype(str), prefix="cat").astype(int)
    return pd.concat([df, cats], axis=1), amt_stats
